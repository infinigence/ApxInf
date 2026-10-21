use apxinf_core::{DType, Error, Result, Shape, Tensor};
use apxinf_cuda_new::ops::{gemm, GemmArgs, GemmPolicy, WeightVersion};
use apxinf_cuda_new::{pointwise, PointwiseArgs, PointwiseSemantic};
use apxinf_cuda_new::{CudaBuffer, CudaContext, PreparedUnfold3d, Unfold3dArgs, Unfold3dGeometry};

use super::super::weights::Conv;

/// Stable per-invocation lowering resources. The shared column buffer is
/// reused between tiles, with all launches ordered on the model stream.
pub(super) struct ConvWorkspace {
    tiles: Vec<Tile>,
    pub output: Tensor,
}

struct Tile {
    unfold: PreparedUnfold3d,
    columns: Tensor,
    destination: Tensor,
    output: Tensor,
}

impl ConvWorkspace {
    pub fn prepare(
        ctx: &CudaContext,
        weights: &Conv,
        input: &Tensor,
        history: Option<&Tensor>,
        stride: [usize; 3],
        padding_before: [usize; 3],
        padding_after: [usize; 3],
    ) -> Result<Self> {
        let dims = input.shape().dims();
        if dims.len() != 4 || dims[3] != weights.input_channels || dims.contains(&0) {
            return Err(Error::Other(
                "Cosmos VAE convolution input geometry mismatch".into(),
            ));
        }
        let mut output_shape = vec![0; 4];
        for axis in 0..3 {
            let padded = dims[axis]
                + padding_before[axis]
                + padding_after[axis]
                + if axis == 0 {
                    history.map_or(0, |h| h.shape().dims()[0])
                } else {
                    0
                };
            if stride[axis] == 0 || padded < weights.kernel[axis] {
                return Err(Error::Other(
                    "Cosmos VAE convolution padding/stride mismatch".into(),
                ));
            }
            output_shape[axis] = (padded - weights.kernel[axis]) / stride[axis] + 1;
        }
        output_shape[3] = weights.output_channels;
        let rows: usize = output_shape[..3].iter().product();
        let tile_rows = rows.min(4096);
        let k = weights.input_channels * weights.kernel.iter().product::<usize>();
        let columns_buffer =
            CudaBuffer::alloc(tile_rows * k * 2, ctx.device_id()).map_err(Error::Cuda)?;
        let output_buffer = CudaBuffer::alloc(rows * weights.output_channels * 2, ctx.device_id())
            .map_err(Error::Cuda)?;
        // Bind every full tile to one GEMM output address. Distinct final
        // destinations would otherwise create one vendor handle/workspace per
        // tile, even though execution is sequential and the weights are shared.
        let tile_buffer =
            CudaBuffer::alloc(tile_rows * weights.output_channels * 2, ctx.device_id())
                .map_err(Error::Cuda)?;
        let output = output_buffer
            .as_tensor(Shape::new(output_shape), DType::BF16)
            .map_err(Error::Cuda)?;
        let mut tiles = Vec::new();
        for start in (0..rows).step_by(tile_rows) {
            let count = tile_rows.min(rows - start);
            let mut columns = columns_buffer
                .view(0, count * k * 2)
                .map_err(Error::Cuda)?
                .as_tensor(Shape::new(vec![count, k]), DType::BF16)
                .map_err(Error::Cuda)?;
            let destination = output_buffer
                .view(
                    start * weights.output_channels * 2,
                    count * weights.output_channels * 2,
                )
                .map_err(Error::Cuda)?
                .as_tensor(
                    Shape::new(vec![count, weights.output_channels]),
                    DType::BF16,
                )
                .map_err(Error::Cuda)?;
            let tile_output = tile_buffer
                .view(0, count * weights.output_channels * 2)
                .map_err(Error::Cuda)?
                .as_tensor(
                    Shape::new(vec![count, weights.output_channels]),
                    DType::BF16,
                )
                .map_err(Error::Cuda)?;
            let unfold = PreparedUnfold3d::new(
                ctx,
                Unfold3dArgs {
                    input,
                    history,
                    row_start: start,
                    out: &mut columns,
                    geometry: Unfold3dGeometry {
                        kernel: weights.kernel,
                        stride,
                        padding_before,
                        padding_after,
                    },
                },
            )?;
            tiles.push(Tile {
                unfold,
                columns,
                destination,
                output: tile_output,
            });
        }
        Ok(Self { tiles, output })
    }

    pub fn forward(
        &mut self,
        ctx: &CudaContext,
        weights: &Conv,
        policy: &GemmPolicy,
    ) -> Result<()> {
        for tile in &mut self.tiles {
            tile.unfold.enqueue()?;
            let mut args = GemmArgs::new(&tile.columns, &weights.linear.weight, &mut tile.output);
            args.policy = policy.clone();
            args.weight_version = Some(WeightVersion::new(1));
            gemm(ctx, args)?;
            if let Some(bias) = &weights.linear.bias {
                // Preserve the convolution-output BF16 rounding before bias.
                let input = tile.output.clone();
                let mut args =
                    PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut tile.output);
                args.bias = Some(bias);
                pointwise(ctx, args)?;
            }
            apxinf_cuda_new::transfers::copy_cuda_to_cuda(ctx, &tile.output, &tile.destination)?;
        }
        Ok(())
    }
}
