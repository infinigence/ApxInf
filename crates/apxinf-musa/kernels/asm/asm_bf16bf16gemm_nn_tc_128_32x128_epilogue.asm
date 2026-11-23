
    .text
    .protected  asm_bf16bf16gemm_nn_tc_128_32x128_epilogue
    .globl  asm_bf16bf16gemm_nn_tc_128_32x128_epilogue
    .p2align    8
    .type   asm_bf16bf16gemm_nn_tc_128_32x128_epilogue,@function
asm_bf16bf16gemm_nn_tc_128_32x128_epilogue:







    ; init stage
[N--D--s]    MOV.LD.LM.B32 R17, LM4
[N--D--s]    MOV.LD.LM.B32 R18, LM5
[N--D--o]    MOV.LD.LM.B32 R0, LM6
[NswD---]    BIT.ROT.LSL SL3, C59, R17
[NwsD---]    BIT.ROT.LSL SL2, C80, R18
[N--D---] FOP.MOV R32, C0
[N--D---] FOP.MOV R33, C0
[NswD---]    INT.ADD R13.s32, SH56.s32, C0.s32
[NwsD---]    BIT.AND.LSL R19, C50, C50, AR0
[NswD---]    BIT.ROT.LSR R20, C45, AR0
[NwsD---]    BIT.ROT.LSL R21, C79, SH31
[N--D---] FOP.MOV R34, C0
[N--D---] FOP.MOV R35, C0
[NswD---]   BIT.AND.LSL R22, C80, C50, AR0
[NwsD---]   BIT.ROT.LSR R23, C45, AR0
[NswD---]   BIT.ROT.LSL R24, C79, SH32
[NwsD---]   BIT.ROT.LSL R12, C79, SH32
[N--D---] FOP.MOV R36, C0
[N--D---] FOP.MOV R37, C0
[NswD---]       INT.ADD R3.s32, SL2.s32, R20.s32
[NwsD---]       INT.MADD R3.s32, SH31.s32, R19.s32, R3.s32
[N--D---] FOP.MOV R38, C0
[N--D---] FOP.MOV R39, C0
[NswD---]       INT.MADD R3.s32, SH38.s32, R3.s32, R0.s32
[NwsD---]       BIT.ROT.LSL R3, C33, R3
[N--D---] FOP.MOV R40, C0
[N--D---] FOP.MOV R41, C0
[NswD---]       INT.MADD R4.s32, SH32.s32, R22.s32, R23.s32
[NwsD---]       INT.ADD  R4.s32, SL3.s32, R4.s32
[NswD---]       INT.MADD R4.s32, SH39.s32, R4.s32, R0.s32
[NwsD---]       BIT.ROT.LSL R4, C33, R4
[NswD---]       INT.ADD R5.s32, R24.s32, R4.s32
[N--D---] FOP.MOV R42, C0
[N--D---] FOP.MOV R43, C0
[N--D---] FOP.MOV R44, C0
[N--D---] FOP.MOV R45, C0
[NwsD---]       INT.ADD R6.s32, R24.s32, R5.s32
[NswD---]       INT.ADD R7.s32, R24.s32, R6.s32
[NwsD---]       INT.ADD R68.s32, SH30.s32, C46.neg.s32
[NswD---]       INT.ADD R105.s32, SH30.s32, C39.neg.s32
[NwsD---]       INT.ADD R106.s32, SH30.s32, C48.neg.s32
[NswD---]       FOP.MOV R1, R23
[NwsD---]       INT.BYP.TL p0, R1.s32, SH30.s32
[NswD---]       INT.BYP.MOVC.PRB R4.s32, R4.s32, SH10.s32, p0
[NwsD---]       INT.BYP.TL p0, R1.s32, R68.s32
[NswD---]       INT.BYP.MOVC.PRB R5.s32, R5.s32, SH10.s32, p0
[NwsD---]       INT.BYP.TL p0, R1.s32, R105.s32
[NswD---]       INT.BYP.MOVC.PRB R6.s32, R6.s32, SH10.s32, p0
[NwsD---]       INT.BYP.TL p0, R1.s32, R106.s32
[NswD---]       INT.BYP.MOVC.PRB R7.s32, R7.s32, SH10.s32, p0
[NwsD---]       INT.ADD R1.s32, R1.s32, SH53.s32
[N--D---] FOP.MOV R46, C0
[N--D---] FOP.MOV R47, C0
[NwwDs--]       DMA.LD.B128 R[85:88], SH[4:5], R3, _, _, _, _,  SH[4:6]
[N--Ds--]       DMA.LD.B512 R[89:104], SH[8:9], R4, _, _, _, _,  SH[8:10]
[N--D---] FOP.MOV R48, C0
[N--D---] FOP.MOV R49, C0
[NswD---]       BIT.AND.LSR R28, C33, C50, AR0
[NwsD---]       BIT.ROT.LSL R28, C33, R28
[NswD---]       BIT.ROT.LSR R25, C79, AR0
[NwsD---]       INT.ADD R28.s32, R28.s32, R25.s32
[NswD---]       BIT.ROT.LSL R28, C46, R28
[N--D---] FOP.MOV R50, C0
[N--D---] FOP.MOV R51, C0
[NwsD---]       BIT.AND.LSL R29, C59, C33, AR0
[N--D---] FOP.MOV R52, C0
[N--D---] FOP.MOV R53, C0
[NswD---]       BIT.AND.LSR R31, C45, SH55, AR0
[NwsD---]       BIT.ROT.LSL R31, C50, R31
[N--D---] FOP.MOV R54, C0
[N--D---] FOP.MOV R55, C0
[NswD---]       INT.ADD R8.s32, R28.s32, R29.s32
[NwsD---]       INT.ADD R8.s32, R8.s32, R31.s32
[NswD---]       BIT.ROT.LSR R8, C33, R8
[N--D---] FOP.MOV R56, C0
[N--D---] FOP.MOV R57, C0
[NwsD---]       BIT.ROT.LSR R26, C79, AR0
[NswD---]       BIT.ROT.LSL R26, SH52, R26
[NwsD---]       BIT.AND.LSL R27, SH51, C50, AR0
[N--D---] FOP.MOV R58, C0
[N--D---] FOP.MOV R59, C0
[NswD---]       BIT.AND.LSR R31, C45, SH55, AR0
[NwsD---]       BIT.ROT.LSL R31, C61, R31
[NswD---]       INT.ADD R9.s32, R26.s32, R27.s32
[NwsD---]       INT.ADD R9.s32, R9.s32, R31.s32
[NswD---]       BIT.ROT.LSR R9, C33, R9
[N--D---] FOP.MOV R60, C0
[N--D---] FOP.MOV R61, C0
[NwsD---]       BIT.ROT.LSR R10, C0, AR0
[NswD---]       BIT.ROT.LSL R11, C0, AR0
[NwsD---]       INT.ADD SL1.s32, SH30.s32, SH53.neg.s32
[N--Dw--] FOP.MOV R62, C0
[N-sD---] FOP.MOV R63, C0





[N-wD-s-]       LDMA.WR.B128.BST1 R[85:88], R8, X4, 128 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[89:92], R9, X4, 2176 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[93:96], R9, X4, 2192 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[97:100], R9, X4, 2688 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[101:104], R9, X4, 2704 , BURST_STRIDE=16
[N-wD---]       INT.ADD R3.s32, R13.s32, R3.s32
[N--D---]       INT.ADD R4.s32, R12.s32, R4.s32
[N--D---]       INT.ADD R5.s32, R12.s32, R5.s32
[N--D---]       INT.ADD R6.s32, R12.s32, R6.s32
[N-sD-w-]       INT.ADD R7.s32, R12.s32, R7.s32
[NswD---]       INT.BYP.TL p0, R1.s32, SH30.s32
[NwsD---]       INT.BYP.MOVC.PRB R4.s32, R4.s32, SH10.s32, p0
[NswD---]       INT.BYP.TL p0, R1.s32, R68.s32
[NwsD---]       INT.BYP.MOVC.PRB R5.s32, R5.s32, SH10.s32, p0
[NswD---]       INT.BYP.TL p0, R1.s32, R105.s32
[NwsD---]       INT.BYP.MOVC.PRB R6.s32, R6.s32, SH10.s32, p0
[NswD---]       INT.BYP.TL p0, R1.s32, R106.s32
[NwsD---]       INT.BYP.MOVC.PRB R7.s32, R7.s32, SH10.s32, p0
[NswD---]       INT.ADD R1.s32, R1.s32, SH53.s32
[NwwD-s-]       LDMA.RD.B32.BST2  R[64:65], R10, X4, 128, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[69:72], R11, X4, 2176, BURST_STRIDE=512
[N--D-o-]       LDMA.RD.B32.BST4  R[73:76], R11, X4, 4224, BURST_STRIDE=512
[NswD--w]       INT.BYP.TG p0, SL1.s32, C0.s32
[Nw-D---]       CTRL.BR.ALLP0F LAST_LOOP


MAIN_LOOP:

[NwwDs--]       DMA.LD.B128 R[85:88], SH[4:5], R3, _, _, _, _,  SH[4:6]
[N--Ds--]       DMA.LD.B512 R[89:104], SH[8:9], R4, _, _, _, _,  SH[8:10]
[N-wD---] MMA.323216 R32[4:2][2:1].f32, R64[2:1][1:1].bf16.row, R69[1:1][2:1].bf16.row, R32[4:2][2:1].f32
[N--D---] MMA.323216 R40[4:2][2:1].f32, R64[2:1][1:1].bf16.row, R71[1:1][2:1].bf16.row, R40[4:2][2:1].f32
[NwwD--s]       LDMA.RD.B32.BST2  R[66:67], R10, X4, 1152, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[77:80], R11, X4, 6272, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[81:84], R11, X4, 8320, BURST_STRIDE=512
[N--D---] MMA.323216 R48[4:2][2:1].f32, R64[2:1][1:1].bf16.row, R73[1:1][2:1].bf16.row, R48[4:2][2:1].f32
[Ns-Dw-w] MMA.323216 R56[4:2][2:1].f32, R64[2:1][1:1].bf16.row, R75[1:1][2:1].bf16.row, R56[4:2][2:1].f32
[N-wD-s-]       LDMA.WR.B128.BST1 R[85:88], R8, X4, 128 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[89:92], R9, X4, 2176 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[93:96], R9, X4, 2192 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[97:100], R9, X4, 2688 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[101:104], R9, X4, 2704 , BURST_STRIDE=16
[N--D---]       INT.ADD R3.s32, R13.s32, R3.s32
[N--D---]       INT.ADD R4.s32, R12.s32, R4.s32
[N--D---]       INT.ADD R5.s32, R12.s32, R5.s32
[N--D---]       INT.ADD R6.s32, R12.s32, R6.s32
[N-sD---]       INT.ADD R7.s32, R12.s32, R7.s32
[N--D---] MMA.323216 R32[4:2][2:1].f32, R66[2:1][1:1].bf16.row, R77[1:1][2:1].bf16.row, R32[4:2][2:1].f32
[NswD---]       INT.BYP.TL p0, R1.s32, SH30.s32
[NwsD---]       INT.BYP.MOVC.PRB R4.s32, R4.s32, SH10.s32, p0
[NswD---]       INT.BYP.TL p0, R1.s32, R68.s32
[NwsD---]       INT.BYP.MOVC.PRB R5.s32, R5.s32, SH10.s32, p0
[NswD---]       INT.BYP.TL p0, R1.s32, R105.s32
[NwsD---]       INT.BYP.MOVC.PRB R6.s32, R6.s32, SH10.s32, p0
[NswD---]       INT.BYP.TL p0, R1.s32, R106.s32
[NwsD---]       INT.BYP.MOVC.PRB R7.s32, R7.s32, SH10.s32, p0
[NswD---]       INT.ADD R1.s32, R1.s32, SH53.s32
[N--D---] MMA.323216 R40[4:2][2:1].f32, R66[2:1][1:1].bf16.row, R79[1:1][2:1].bf16.row, R40[4:2][2:1].f32
[NwwD--s]       LDMA.RD.B32.BST2  R[64:65], R10, X4, 128, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[69:72], R11, X4, 2176, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[73:76], R11, X4, 4224, BURST_STRIDE=512
[N--D-w-] MMA.323216 R48[4:2][2:1].f32, R66[2:1][1:1].bf16.row, R81[1:1][2:1].bf16.row, R48[4:2][2:1].f32
[N-sD--w] MMA.323216 R56[4:2][2:1].f32, R66[2:1][1:1].bf16.row, R83[1:1][2:1].bf16.row, R56[4:2][2:1].f32
[Ns-D---]    INT.ADD SL1.s32, SL1.s32, SH53.neg.s32
[NwsD---]    INT.BYP.TG p0, SL1.s32, C0.s32
[N-wD---]    CTRL.BR.ANYP0T MAIN_LOOP


LAST_LOOP:
[N-wD---] MMA.323216 R32[4:2][2:1].f32, R64[2:1][1:1].bf16.row, R69[1:1][2:1].bf16.row, R32[4:2][2:1].f32
[NswD---]       BIT.ROT.LSR R26, C79, AR0
[NwsD---]       BIT.ROT.LSL R26, C45, R26
[NswD---]       BIT.AND.LSR R3, C50, C78, AR0
[NwsD---]       INT.ADD R26.s32, R3.s32, R26.s32
[NswD---]       INT.ADD R26.s32, SL2.s32, R26.s32
[NwsD---]       BIT.AND.LSR R27, C80, SH55, AR0
[N--D---] MMA.323216 R40[4:2][2:1].f32, R64[2:1][1:1].bf16.row, R71[1:1][2:1].bf16.row, R40[4:2][2:1].f32
[N-wD--s]       LDMA.RD.B32.BST2  R[66:67], R10, X4, 1152, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[77:80], R11, X4, 6272, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[81:84], R11, X4, 8320, BURST_STRIDE=512
[N--D---] MMA.323216 R48[4:2][2:1].f32, R64[2:1][1:1].bf16.row, R73[1:1][2:1].bf16.row, R48[4:2][2:1].f32
[NswD---]       BIT.ROT.LSL R27, C50, R27
[NwsD---]       BIT.LUT.AND R3, C59, AR0
[NswD---]       INT.ADD R27.s32, R3.s32, R27.s32
[NwsD---]       INT.ADD R27.s32, SL3.s32, R27.s32
[Ns-D--w] MMA.323216 R56[4:2][2:1].f32, R64[2:1][1:1].bf16.row, R75[1:1][2:1].bf16.row, R56[4:2][2:1].f32
[N-wD---] MMA.323216 R32[4:2][2:1].f32, R66[2:1][1:1].bf16.row, R77[1:1][2:1].bf16.row, R32[4:2][2:1].f32
[NswD---]       FOP.MOV R12, C0
[N--D---]       FOP.MOV R13, C0
[N--D---]       FOP.MOV R14, C0
[N--D---]       FOP.MOV R15, C0
[N--D---]       FOP.MOV R16, C0
[N--D---]       FOP.MOV R17, C0
[N--D---]       FOP.MOV R18, C0
[NwsD---]       FOP.MOV R19, C0
[N--D---] MMA.323216 R40[4:2][2:1].f32, R66[2:1][1:1].bf16.row, R79[1:1][2:1].bf16.row, R40[4:2][2:1].f32
[NwsD---]        BIT.LUT.OR.TZ p0, SH13, SH12
[NswD---]        INT.BYP.TE.OP p0, SH36.s32, C0.s32
[NwwD---]        CTRL.BR.ANYP0T BB0_BIAS_END
[NwsD---]    BIT.ROT.LSL R12, C33, R27
[N-wD---]    INT.ADD R13.s32, SH53.s32, R12.s32
[N--D---]    INT.ADD R14.s32, SH56.s32, R12.s32
[N--D---]    INT.ADD R15.s32, SH58.s32, R12.s32
[N--D---]    INT.ADD R16.s32, SH60.s32, R12.s32
[N--D---]    INT.ADD R17.s32, SH61.s32, R12.s32
[N--D---]    INT.ADD R18.s32, SH62.s32, R12.s32
[N-sD---]    INT.ADD R19.s32, SH63.s32, R12.s32
[N-wD-s-]        DMA.LD.B16 R12.l, SH[12:13], R12, _, _, _, _,  SH[12:14]
[N--D-s-]        DMA.LD.B16 R13.l, SH[12:13], R13, _, _, _, _,  SH[12:14]
[N--D-s-]        DMA.LD.B16 R14.l, SH[12:13], R14, _, _, _, _,  SH[12:14]
[N--D-s-]        DMA.LD.B16 R15.l, SH[12:13], R15, _, _, _, _,  SH[12:14]
[N--D-s-]        DMA.LD.B16 R16.l, SH[12:13], R16, _, _, _, _,  SH[12:14]
[N--D-s-]        DMA.LD.B16 R17.l, SH[12:13], R17, _, _, _, _,  SH[12:14]
[N--D-s-]        DMA.LD.B16 R18.l, SH[12:13], R18, _, _, _, _,  SH[12:14]
[N--D-s-]        DMA.LD.B16 R19.l, SH[12:13], R19, _, _, _, _,  SH[12:14]
BB0_BIAS_END:
[N--D---] MMA.323216 R48[4:2][2:1].f32, R66[2:1][1:1].bf16.row, R81[1:1][2:1].bf16.row, R48[4:2][2:1].f32
[N-sDww-] MMA.323216 R56[4:2][2:1].f32, R66[2:1][1:1].bf16.row, R83[1:1][2:1].bf16.row, R56[4:2][2:1].f32
; core loop stage end

; debug use



[NswD---]       INT.MADD R28.s32, SH33.s32, R27.s32, R26.s32
[NwsD---]       INT.MADD R28.s32, SH40.s32, R28.s32, R0.s32
[NswD---]       BIT.ROT.LSL R28, C33, R28
[NwsD---]       FOP.MOV R30, SH33
[NswD---]       INT.ADD R3.s32, SH28.s32, R26.neg.s32
[NwsD---]       INT.ADD R4.s32, SH29.s32, R27.neg.s32
[NswD---]       INT.ADD R107.s32, C0.s32, R28.s32
[NwsD---]       INT.ADD R108.s32, SH53.s32, R28.s32
[NswD---]       INT.ADD R109.s32, SH56.s32, R28.s32
[NwsD---]       INT.ADD R110.s32, SH58.s32, R28.s32
[NswD---]       INT.ADD R111.s32, SH60.s32, R28.s32
[NwsD---]       INT.ADD R112.s32, SH61.s32, R28.s32
[NswD---]       INT.MUL R29.s32, C39.s32,  R30.s32
[NwsD---]       INT.ADD R113.s32, SH62.s32, R28.s32
[NswD---]       INT.ADD R114.s32, SH63.s32, R28.s32
[NwsD---]       INT.BYP.TLE p0, R3.s32, C48.s32
[NswD---]       INT.BYP.TLE.OP p0, R4.s32, SH59.s32
[NwwD---]        CTRL.BR.ANYP0T B_STORE_MASK




[N-sD---]        FOP.CMP.TE p0 C0, SH35
[NwwD---]        CTRL.BR.ANYP0T STORE_NORMAL
STORE_NORMAL_WITH_BETA:
[N-wD---] FOP.MOV R115, R107
[N--D---] FOP.MOV R116, R108
[N--D---] FOP.MOV R117, R109
[N--D---] FOP.MOV R118, R110
[N--D---] FOP.MOV R119, R111
[N--D---] FOP.MOV R120, R112
[N--D---] FOP.MOV R121, R113
[N-sD---] FOP.MOV R122, R114
[N-wDs--]       DMA.LD.B16 R123.l, SH[16:17], R115, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R124.l, SH[16:17], R116, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R131.l, SH[16:17], R117, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R132.l, SH[16:17], R118, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R139.l, SH[16:17], R119, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R140.l, SH[16:17], R120, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R147.l, SH[16:17], R121, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R148.l, SH[16:17], R122, _, _, _, _,  SH[16:18]
[N-wDw--] CTRL.NOP
[N-wD---] INT.ADD R115.s32, R29.s32, R115.s32
[N--D---] INT.ADD R116.s32, R29.s32, R116.s32
[N--D---] INT.ADD R117.s32, R29.s32, R117.s32
[N--D---] INT.ADD R118.s32, R29.s32, R118.s32
[N--D---] INT.ADD R119.s32, R29.s32, R119.s32
[N--D---] INT.ADD R120.s32, R29.s32, R120.s32
[N--D---] INT.ADD R121.s32, R29.s32, R121.s32
[N-sD---] INT.ADD R122.s32, R29.s32, R122.s32
[N-wDs--]       DMA.LD.B16 R125.l, SH[16:17], R115, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R126.l, SH[16:17], R116, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R133.l, SH[16:17], R117, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R134.l, SH[16:17], R118, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R141.l, SH[16:17], R119, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R142.l, SH[16:17], R120, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R149.l, SH[16:17], R121, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R150.l, SH[16:17], R122, _, _, _, _,  SH[16:18]
[N-wDw--] CTRL.NOP
[N-wD---] INT.ADD R115.s32, R29.s32, R115.s32
[N--D---] INT.ADD R116.s32, R29.s32, R116.s32
[N--D---] INT.ADD R117.s32, R29.s32, R117.s32
[N--D---] INT.ADD R118.s32, R29.s32, R118.s32
[N--D---] INT.ADD R119.s32, R29.s32, R119.s32
[N--D---] INT.ADD R120.s32, R29.s32, R120.s32
[N--D---] INT.ADD R121.s32, R29.s32, R121.s32
[N-sD---] INT.ADD R122.s32, R29.s32, R122.s32
[N-wDs--]       DMA.LD.B16 R127.l, SH[16:17], R115, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R128.l, SH[16:17], R116, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R135.l, SH[16:17], R117, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R136.l, SH[16:17], R118, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R143.l, SH[16:17], R119, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R144.l, SH[16:17], R120, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R151.l, SH[16:17], R121, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R152.l, SH[16:17], R122, _, _, _, _,  SH[16:18]
[N-wDw--] CTRL.NOP
[N-wD---] INT.ADD R115.s32, R29.s32, R115.s32
[N--D---] INT.ADD R116.s32, R29.s32, R116.s32
[N--D---] INT.ADD R117.s32, R29.s32, R117.s32
[N--D---] INT.ADD R118.s32, R29.s32, R118.s32
[N--D---] INT.ADD R119.s32, R29.s32, R119.s32
[N--D---] INT.ADD R120.s32, R29.s32, R120.s32
[N--D---] INT.ADD R121.s32, R29.s32, R121.s32
[N-sD---] INT.ADD R122.s32, R29.s32, R122.s32
[N-wDs--]       DMA.LD.B16 R129.l, SH[16:17], R115, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R130.l, SH[16:17], R116, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R137.l, SH[16:17], R117, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R138.l, SH[16:17], R118, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R145.l, SH[16:17], R119, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R146.l, SH[16:17], R120, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R153.l, SH[16:17], R121, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R154.l, SH[16:17], R122, _, _, _, _,  SH[16:18]
[N-wDw--] CTRL.NOP
[N-wD---] INT.ADD R115.s32, R29.s32, R115.s32
[N--D---] INT.ADD R116.s32, R29.s32, R116.s32
[N--D---] INT.ADD R117.s32, R29.s32, R117.s32
[N--D---] INT.ADD R118.s32, R29.s32, R118.s32
[N--D---] INT.ADD R119.s32, R29.s32, R119.s32
[N--D---] INT.ADD R120.s32, R29.s32, R120.s32
[N--D---] INT.ADD R121.s32, R29.s32, R121.s32
[N-sD---] INT.ADD R122.s32, R29.s32, R122.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R32, SH34, R32, R68
[N--D---] FOP.FMA R33, SH34, R33, R69
[N--D---] FOP.FMA R32.l.bf16, R123.l.bf16, SH35, R32
[N--D---] FOP.FMA R33.l.bf16, R124.l.bf16, SH35, R33
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R40, SH34, R40, R70
[N--D---] FOP.FMA R41, SH34, R41, R71
[N--D---] FOP.FMA R40.l.bf16, R131.l.bf16, SH35, R40
[N--D---] FOP.FMA R41.l.bf16, R132.l.bf16, SH35, R41
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R48, SH34, R48, R72
[N--D---] FOP.FMA R49, SH34, R49, R73
[N--D---] FOP.FMA R48.l.bf16, R139.l.bf16, SH35, R48
[N--D---] FOP.FMA R49.l.bf16, R140.l.bf16, SH35, R49
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R56, SH34, R56, R74
[N--D---] FOP.FMA R57, SH34, R57, R75
[N--D---] FOP.FMA R56.l.bf16, R147.l.bf16, SH35, R56
[N-sD---] FOP.FMA R57.l.bf16, R148.l.bf16, SH35, R57
[N-wD---]       DMA.ST.B16 R32.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R33.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R40.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R41.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R48.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R49.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R56.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R57.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R34, SH34, R34, R68
[N--D---] FOP.FMA R35, SH34, R35, R69
[N--D---] FOP.FMA R34.l.bf16, R125.l.bf16, SH35, R34
[N--D---] FOP.FMA R35.l.bf16, R126.l.bf16, SH35, R35
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R42, SH34, R42, R70
[N--D---] FOP.FMA R43, SH34, R43, R71
[N--D---] FOP.FMA R42.l.bf16, R133.l.bf16, SH35, R42
[N--D---] FOP.FMA R43.l.bf16, R134.l.bf16, SH35, R43
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R50, SH34, R50, R72
[N--D---] FOP.FMA R51, SH34, R51, R73
[N--D---] FOP.FMA R50.l.bf16, R141.l.bf16, SH35, R50
[N--D---] FOP.FMA R51.l.bf16, R142.l.bf16, SH35, R51
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R58, SH34, R58, R74
[N--D---] FOP.FMA R59, SH34, R59, R75
[N--D---] FOP.FMA R58.l.bf16, R149.l.bf16, SH35, R58
[N-sD---] FOP.FMA R59.l.bf16, R150.l.bf16, SH35, R59
[N-wD---]       DMA.ST.B16 R34.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R35.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R42.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R43.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R50.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R51.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R58.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R59.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R36, SH34, R36, R68
[N--D---] FOP.FMA R37, SH34, R37, R69
[N--D---] FOP.FMA R36.l.bf16, R127.l.bf16, SH35, R36
[N--D---] FOP.FMA R37.l.bf16, R128.l.bf16, SH35, R37
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R44, SH34, R44, R70
[N--D---] FOP.FMA R45, SH34, R45, R71
[N--D---] FOP.FMA R44.l.bf16, R135.l.bf16, SH35, R44
[N--D---] FOP.FMA R45.l.bf16, R136.l.bf16, SH35, R45
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R52, SH34, R52, R72
[N--D---] FOP.FMA R53, SH34, R53, R73
[N--D---] FOP.FMA R52.l.bf16, R143.l.bf16, SH35, R52
[N--D---] FOP.FMA R53.l.bf16, R144.l.bf16, SH35, R53
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R60, SH34, R60, R74
[N--D---] FOP.FMA R61, SH34, R61, R75
[N--D---] FOP.FMA R60.l.bf16, R151.l.bf16, SH35, R60
[N-sD---] FOP.FMA R61.l.bf16, R152.l.bf16, SH35, R61
[N-wD---]       DMA.ST.B16 R36.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R37.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R44.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R45.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R52.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R53.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R60.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R61.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R38, SH34, R38, R68
[N--D---] FOP.FMA R39, SH34, R39, R69
[N--D---] FOP.FMA R38.l.bf16, R129.l.bf16, SH35, R38
[N--D---] FOP.FMA R39.l.bf16, R130.l.bf16, SH35, R39
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R46, SH34, R46, R70
[N--D---] FOP.FMA R47, SH34, R47, R71
[N--D---] FOP.FMA R46.l.bf16, R137.l.bf16, SH35, R46
[N--D---] FOP.FMA R47.l.bf16, R138.l.bf16, SH35, R47
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R54, SH34, R54, R72
[N--D---] FOP.FMA R55, SH34, R55, R73
[N--D---] FOP.FMA R54.l.bf16, R145.l.bf16, SH35, R54
[N--D---] FOP.FMA R55.l.bf16, R146.l.bf16, SH35, R55
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R62, SH34, R62, R74
[N--D---] FOP.FMA R63, SH34, R63, R75
[N--D---] FOP.FMA R62.l.bf16, R153.l.bf16, SH35, R62
[N-sD---] FOP.FMA R63.l.bf16, R154.l.bf16, SH35, R63
[N-wD---]       DMA.ST.B16 R38.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R39.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R46.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R47.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R54.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R55.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R62.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R63.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[Nw-D---]        CTRL.NOP
[N-wD---]        CTRL.END
STORE_NORMAL:
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R32.l.bf16, SH34, R32, R68
[N--D---] FOP.FMA R33.l.bf16, SH34, R33, R69
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R40.l.bf16, SH34, R40, R70
[N--D---] FOP.FMA R41.l.bf16, SH34, R41, R71
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R48.l.bf16, SH34, R48, R72
[N--D---] FOP.FMA R49.l.bf16, SH34, R49, R73
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R56.l.bf16, SH34, R56, R74
[N-sD---] FOP.FMA R57.l.bf16, SH34, R57, R75
[N-wD---]       DMA.ST.B16 R32.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R33.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R40.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R41.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R48.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R49.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R56.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R57.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R34.l.bf16, SH34, R34, R68
[N--D---] FOP.FMA R35.l.bf16, SH34, R35, R69
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R42.l.bf16, SH34, R42, R70
[N--D---] FOP.FMA R43.l.bf16, SH34, R43, R71
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R50.l.bf16, SH34, R50, R72
[N--D---] FOP.FMA R51.l.bf16, SH34, R51, R73
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R58.l.bf16, SH34, R58, R74
[N-sD---] FOP.FMA R59.l.bf16, SH34, R59, R75
[N-wD---]       DMA.ST.B16 R34.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R35.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R42.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R43.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R50.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R51.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R58.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R59.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R36.l.bf16, SH34, R36, R68
[N--D---] FOP.FMA R37.l.bf16, SH34, R37, R69
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R44.l.bf16, SH34, R44, R70
[N--D---] FOP.FMA R45.l.bf16, SH34, R45, R71
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R52.l.bf16, SH34, R52, R72
[N--D---] FOP.FMA R53.l.bf16, SH34, R53, R73
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R60.l.bf16, SH34, R60, R74
[N-sD---] FOP.FMA R61.l.bf16, SH34, R61, R75
[N-wD---]       DMA.ST.B16 R36.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R37.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R44.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R45.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R52.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R53.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R60.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R61.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R38.l.bf16, SH34, R38, R68
[N--D---] FOP.FMA R39.l.bf16, SH34, R39, R69
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R46.l.bf16, SH34, R46, R70
[N--D---] FOP.FMA R47.l.bf16, SH34, R47, R71
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R54.l.bf16, SH34, R54, R72
[N--D---] FOP.FMA R55.l.bf16, SH34, R55, R73
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R62.l.bf16, SH34, R62, R74
[N-sD---] FOP.FMA R63.l.bf16, SH34, R63, R75
[N-wD---]       DMA.ST.B16 R38.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R39.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R46.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R47.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R54.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R55.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R62.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R63.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[Nw-D---]        CTRL.NOP
[N-wD---]        CTRL.END



  ; [NwsD---]       FOP.MUL R25, SH64, R25
  ; [NwsD---]       FOP.MUL R25, C81, R25
  ; [NswD---]       BIT.ROT.LSL R31, C45, AR0
  ; [NwsD---] MOV.B32  R11, SH0[R10]
  ; [NwsD---]       DMA.ST.B32 R3, SH[0:1], R31
  [Nw-D---]   CTRL.NOP
  [N-wD---]   CTRL.END


B_STORE_MASK:
[N-sD---]        FOP.CMP.TE p0 C0, SH35
[NwwD---]        CTRL.BR.ANYP0T STORE_MASK
STORE_MASK_WITH_BETA:
[NwwD---] FOP.MOV R115, R107
[N--D---] FOP.MOV R116, R108
[N--D---] FOP.MOV R117, R109
[N--D---] FOP.MOV R118, R110
[N--D---] FOP.MOV R119, R111
[N--D---] FOP.MOV R120, R112
[N--D---] FOP.MOV R121, R113
[N-sD---] FOP.MOV R122, R114
[N-wDs--]       DMA.LD.B16 R123.l, SH[16:17], R115, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R124.l, SH[16:17], R116, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R131.l, SH[16:17], R117, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R132.l, SH[16:17], R118, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R139.l, SH[16:17], R119, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R140.l, SH[16:17], R120, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R147.l, SH[16:17], R121, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R148.l, SH[16:17], R122, _, _, _, _,  SH[16:18]
[N-wDw--] CTRL.NOP
[N-wD---] INT.ADD R115.s32, R29.s32, R115.s32
[N--D---] INT.ADD R116.s32, R29.s32, R116.s32
[N--D---] INT.ADD R117.s32, R29.s32, R117.s32
[N--D---] INT.ADD R118.s32, R29.s32, R118.s32
[N--D---] INT.ADD R119.s32, R29.s32, R119.s32
[N--D---] INT.ADD R120.s32, R29.s32, R120.s32
[N--D---] INT.ADD R121.s32, R29.s32, R121.s32
[N-sD---] INT.ADD R122.s32, R29.s32, R122.s32
[N-wDs--]       DMA.LD.B16 R125.l, SH[16:17], R115, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R126.l, SH[16:17], R116, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R133.l, SH[16:17], R117, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R134.l, SH[16:17], R118, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R141.l, SH[16:17], R119, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R142.l, SH[16:17], R120, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R149.l, SH[16:17], R121, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R150.l, SH[16:17], R122, _, _, _, _,  SH[16:18]
[N-wDw--] CTRL.NOP
[N-wD---] INT.ADD R115.s32, R29.s32, R115.s32
[N--D---] INT.ADD R116.s32, R29.s32, R116.s32
[N--D---] INT.ADD R117.s32, R29.s32, R117.s32
[N--D---] INT.ADD R118.s32, R29.s32, R118.s32
[N--D---] INT.ADD R119.s32, R29.s32, R119.s32
[N--D---] INT.ADD R120.s32, R29.s32, R120.s32
[N--D---] INT.ADD R121.s32, R29.s32, R121.s32
[N-sD---] INT.ADD R122.s32, R29.s32, R122.s32
[N-wDs--]       DMA.LD.B16 R127.l, SH[16:17], R115, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R128.l, SH[16:17], R116, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R135.l, SH[16:17], R117, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R136.l, SH[16:17], R118, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R143.l, SH[16:17], R119, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R144.l, SH[16:17], R120, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R151.l, SH[16:17], R121, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R152.l, SH[16:17], R122, _, _, _, _,  SH[16:18]
[N-wDw--] CTRL.NOP
[N-wD---] INT.ADD R115.s32, R29.s32, R115.s32
[N--D---] INT.ADD R116.s32, R29.s32, R116.s32
[N--D---] INT.ADD R117.s32, R29.s32, R117.s32
[N--D---] INT.ADD R118.s32, R29.s32, R118.s32
[N--D---] INT.ADD R119.s32, R29.s32, R119.s32
[N--D---] INT.ADD R120.s32, R29.s32, R120.s32
[N--D---] INT.ADD R121.s32, R29.s32, R121.s32
[N-sD---] INT.ADD R122.s32, R29.s32, R122.s32
[N-wDs--]       DMA.LD.B16 R129.l, SH[16:17], R115, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R130.l, SH[16:17], R116, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R137.l, SH[16:17], R117, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R138.l, SH[16:17], R118, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R145.l, SH[16:17], R119, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R146.l, SH[16:17], R120, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R153.l, SH[16:17], R121, _, _, _, _,  SH[16:18]
[N--Ds--]       DMA.LD.B16 R154.l, SH[16:17], R122, _, _, _, _,  SH[16:18]
[N-wDw--] CTRL.NOP
[N-wD---] INT.ADD R115.s32, R29.s32, R115.s32
[N--D---] INT.ADD R116.s32, R29.s32, R116.s32
[N--D---] INT.ADD R117.s32, R29.s32, R117.s32
[N--D---] INT.ADD R118.s32, R29.s32, R118.s32
[N--D---] INT.ADD R119.s32, R29.s32, R119.s32
[N--D---] INT.ADD R120.s32, R29.s32, R120.s32
[N--D---] INT.ADD R121.s32, R29.s32, R121.s32
[N-sD---] INT.ADD R122.s32, R29.s32, R122.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R32, SH34, R32, R68
[N--D---] FOP.FMA R33, SH34, R33, R69
[N--D---] FOP.FMA R32.l.bf16, R123.l.bf16, SH35, R32
[N--D---] FOP.FMA R33.l.bf16, R124.l.bf16, SH35, R33
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R40, SH34, R40, R70
[N--D---] FOP.FMA R41, SH34, R41, R71
[N--D---] FOP.FMA R40.l.bf16, R131.l.bf16, SH35, R40
[N--D---] FOP.FMA R41.l.bf16, R132.l.bf16, SH35, R41
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R48, SH34, R48, R72
[N--D---] FOP.FMA R49, SH34, R49, R73
[N--D---] FOP.FMA R48.l.bf16, R139.l.bf16, SH35, R48
[N--D---] FOP.FMA R49.l.bf16, R140.l.bf16, SH35, R49
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R56, SH34, R56, R74
[N--D---] FOP.FMA R57, SH34, R57, R75
[N--D---] FOP.FMA R56.l.bf16, R147.l.bf16, SH35, R56
[N-sD---] FOP.FMA R57.l.bf16, R148.l.bf16, SH35, R57
[NswD---]       INT.BYP.TG p0, R3.s32, C0.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R32.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R33.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R40.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH54.s32
[NswD---]   P0  DMA.ST.B16 R41.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH56.s32
[NswD---]   P0  DMA.ST.B16 R48.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH57.s32
[NswD---]   P0  DMA.ST.B16 R49.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH58.s32
[NswD---]   P0  DMA.ST.B16 R56.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH59.s32
[NswD---]   P0  DMA.ST.B16 R57.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R34, SH34, R34, R68
[N--D---] FOP.FMA R35, SH34, R35, R69
[N--D---] FOP.FMA R34.l.bf16, R125.l.bf16, SH35, R34
[N--D---] FOP.FMA R35.l.bf16, R126.l.bf16, SH35, R35
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R42, SH34, R42, R70
[N--D---] FOP.FMA R43, SH34, R43, R71
[N--D---] FOP.FMA R42.l.bf16, R133.l.bf16, SH35, R42
[N--D---] FOP.FMA R43.l.bf16, R134.l.bf16, SH35, R43
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R50, SH34, R50, R72
[N--D---] FOP.FMA R51, SH34, R51, R73
[N--D---] FOP.FMA R50.l.bf16, R141.l.bf16, SH35, R50
[N--D---] FOP.FMA R51.l.bf16, R142.l.bf16, SH35, R51
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R58, SH34, R58, R74
[N--D---] FOP.FMA R59, SH34, R59, R75
[N--D---] FOP.FMA R58.l.bf16, R149.l.bf16, SH35, R58
[N-sD---] FOP.FMA R59.l.bf16, R150.l.bf16, SH35, R59
[NwsD---]       INT.BYP.TG p0, R3.s32, C46.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R34.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R35.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R42.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH54.s32
[NwsD---]   P0  DMA.ST.B16 R43.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH56.s32
[NwsD---]   P0  DMA.ST.B16 R50.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH57.s32
[NwsD---]   P0  DMA.ST.B16 R51.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH58.s32
[NwsD---]   P0  DMA.ST.B16 R58.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH59.s32
[NwsD---]   P0  DMA.ST.B16 R59.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R36, SH34, R36, R68
[N--D---] FOP.FMA R37, SH34, R37, R69
[N--D---] FOP.FMA R36.l.bf16, R127.l.bf16, SH35, R36
[N--D---] FOP.FMA R37.l.bf16, R128.l.bf16, SH35, R37
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R44, SH34, R44, R70
[N--D---] FOP.FMA R45, SH34, R45, R71
[N--D---] FOP.FMA R44.l.bf16, R135.l.bf16, SH35, R44
[N--D---] FOP.FMA R45.l.bf16, R136.l.bf16, SH35, R45
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R52, SH34, R52, R72
[N--D---] FOP.FMA R53, SH34, R53, R73
[N--D---] FOP.FMA R52.l.bf16, R143.l.bf16, SH35, R52
[N--D---] FOP.FMA R53.l.bf16, R144.l.bf16, SH35, R53
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R60, SH34, R60, R74
[N--D---] FOP.FMA R61, SH34, R61, R75
[N--D---] FOP.FMA R60.l.bf16, R151.l.bf16, SH35, R60
[N-sD---] FOP.FMA R61.l.bf16, R152.l.bf16, SH35, R61
[NswD---]       INT.BYP.TG p0, R3.s32, C39.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R36.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R37.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R44.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH54.s32
[NswD---]   P0  DMA.ST.B16 R45.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH56.s32
[NswD---]   P0  DMA.ST.B16 R52.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH57.s32
[NswD---]   P0  DMA.ST.B16 R53.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH58.s32
[NswD---]   P0  DMA.ST.B16 R60.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH59.s32
[NswD---]   P0  DMA.ST.B16 R61.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R38, SH34, R38, R68
[N--D---] FOP.FMA R39, SH34, R39, R69
[N--D---] FOP.FMA R38.l.bf16, R129.l.bf16, SH35, R38
[N--D---] FOP.FMA R39.l.bf16, R130.l.bf16, SH35, R39
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R46, SH34, R46, R70
[N--D---] FOP.FMA R47, SH34, R47, R71
[N--D---] FOP.FMA R46.l.bf16, R137.l.bf16, SH35, R46
[N--D---] FOP.FMA R47.l.bf16, R138.l.bf16, SH35, R47
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R54, SH34, R54, R72
[N--D---] FOP.FMA R55, SH34, R55, R73
[N--D---] FOP.FMA R54.l.bf16, R145.l.bf16, SH35, R54
[N--D---] FOP.FMA R55.l.bf16, R146.l.bf16, SH35, R55
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R62, SH34, R62, R74
[N--D---] FOP.FMA R63, SH34, R63, R75
[N--D---] FOP.FMA R62.l.bf16, R153.l.bf16, SH35, R62
[N-sD---] FOP.FMA R63.l.bf16, R154.l.bf16, SH35, R63
[NwsD---]       INT.BYP.TG p0, R3.s32, C48.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R38.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R39.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R46.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH54.s32
[NwsD---]   P0  DMA.ST.B16 R47.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH56.s32
[NwsD---]   P0  DMA.ST.B16 R54.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH57.s32
[NwsD---]   P0  DMA.ST.B16 R55.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH58.s32
[NwsD---]   P0  DMA.ST.B16 R62.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH59.s32
[NwsD---]   P0  DMA.ST.B16 R63.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[Nw-D---]        CTRL.NOP
[N-wD---]        CTRL.END
STORE_MASK:
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R32.l.bf16, SH34, R32, R68
[N--D---] FOP.FMA R33.l.bf16, SH34, R33, R69
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R40.l.bf16, SH34, R40, R70
[N--D---] FOP.FMA R41.l.bf16, SH34, R41, R71
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R48.l.bf16, SH34, R48, R72
[N--D---] FOP.FMA R49.l.bf16, SH34, R49, R73
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R56.l.bf16, SH34, R56, R74
[N-sD---] FOP.FMA R57.l.bf16, SH34, R57, R75
[NswD---]       INT.BYP.TG p0, R3.s32, C0.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R32.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R33.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R40.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH54.s32
[NswD---]   P0  DMA.ST.B16 R41.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH56.s32
[NswD---]   P0  DMA.ST.B16 R48.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH57.s32
[NswD---]   P0  DMA.ST.B16 R49.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH58.s32
[NswD---]   P0  DMA.ST.B16 R56.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH59.s32
[NswD---]   P0  DMA.ST.B16 R57.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R34.l.bf16, SH34, R34, R68
[N--D---] FOP.FMA R35.l.bf16, SH34, R35, R69
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R42.l.bf16, SH34, R42, R70
[N--D---] FOP.FMA R43.l.bf16, SH34, R43, R71
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R50.l.bf16, SH34, R50, R72
[N--D---] FOP.FMA R51.l.bf16, SH34, R51, R73
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R58.l.bf16, SH34, R58, R74
[N-sD---] FOP.FMA R59.l.bf16, SH34, R59, R75
[NwsD---]       INT.BYP.TG p0, R3.s32, C46.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R34.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R35.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R42.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH54.s32
[NwsD---]   P0  DMA.ST.B16 R43.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH56.s32
[NwsD---]   P0  DMA.ST.B16 R50.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH57.s32
[NwsD---]   P0  DMA.ST.B16 R51.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH58.s32
[NwsD---]   P0  DMA.ST.B16 R58.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH59.s32
[NwsD---]   P0  DMA.ST.B16 R59.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R36.l.bf16, SH34, R36, R68
[N--D---] FOP.FMA R37.l.bf16, SH34, R37, R69
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R44.l.bf16, SH34, R44, R70
[N--D---] FOP.FMA R45.l.bf16, SH34, R45, R71
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R52.l.bf16, SH34, R52, R72
[N--D---] FOP.FMA R53.l.bf16, SH34, R53, R73
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R60.l.bf16, SH34, R60, R74
[N-sD---] FOP.FMA R61.l.bf16, SH34, R61, R75
[NswD---]       INT.BYP.TG p0, R3.s32, C39.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R36.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R37.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R44.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH54.s32
[NswD---]   P0  DMA.ST.B16 R45.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH56.s32
[NswD---]   P0  DMA.ST.B16 R52.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH57.s32
[NswD---]   P0  DMA.ST.B16 R53.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH58.s32
[NswD---]   P0  DMA.ST.B16 R60.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH59.s32
[NswD---]   P0  DMA.ST.B16 R61.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[N-wD---] FOP.MUL R68, R12.l.bf16, SH36
[N--D---] FOP.MUL R69, R13.l.bf16, SH36
[N--D---] FOP.FMA R38.l.bf16, SH34, R38, R68
[N--D---] FOP.FMA R39.l.bf16, SH34, R39, R69
[N--D---] FOP.MUL R70, R14.l.bf16, SH36
[N--D---] FOP.MUL R71, R15.l.bf16, SH36
[N--D---] FOP.FMA R46.l.bf16, SH34, R46, R70
[N--D---] FOP.FMA R47.l.bf16, SH34, R47, R71
[N--D---] FOP.MUL R72, R16.l.bf16, SH36
[N--D---] FOP.MUL R73, R17.l.bf16, SH36
[N--D---] FOP.FMA R54.l.bf16, SH34, R54, R72
[N--D---] FOP.FMA R55.l.bf16, SH34, R55, R73
[N--D---] FOP.MUL R74, R18.l.bf16, SH36
[N--D---] FOP.MUL R75, R19.l.bf16, SH36
[N--D---] FOP.FMA R62.l.bf16, SH34, R62, R74
[N-sD---] FOP.FMA R63.l.bf16, SH34, R63, R75
[NwsD---]       INT.BYP.TG p0, R3.s32, C48.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R38.l, SH[0:1], R107, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R39.l, SH[0:1], R108, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R46.l, SH[0:1], R109, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH54.s32
[NwsD---]   P0  DMA.ST.B16 R47.l, SH[0:1], R110, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH56.s32
[NwsD---]   P0  DMA.ST.B16 R54.l, SH[0:1], R111, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH57.s32
[NwsD---]   P0  DMA.ST.B16 R55.l, SH[0:1], R112, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH58.s32
[NwsD---]   P0  DMA.ST.B16 R62.l, SH[0:1], R113, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH59.s32
[NwsD---]   P0  DMA.ST.B16 R63.l, SH[0:1], R114, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R107.s32, R29.s32, R107.s32
[N--D---] INT.ADD R108.s32, R29.s32, R108.s32
[N--D---] INT.ADD R109.s32, R29.s32, R109.s32
[N--D---] INT.ADD R110.s32, R29.s32, R110.s32
[N--D---] INT.ADD R111.s32, R29.s32, R111.s32
[N--D---] INT.ADD R112.s32, R29.s32, R112.s32
[N--D---] INT.ADD R113.s32, R29.s32, R113.s32
[N-sD---] INT.ADD R114.s32, R29.s32, R114.s32
[Nw-D---]        CTRL.NOP
[N-wD---]        CTRL.END

[Nw-D---]   CTRL.NOP
[N-wD---]   CTRL.END

.Lfunc_end0:
    .size   asm_bf16bf16gemm_nn_tc_128_32x128_epilogue, .Lfunc_end0-asm_bf16bf16gemm_nn_tc_128_32x128_epilogue




    .type   llvm.mtgpu.kernel.asm_bf16bf16gemm_nn_tc_128_32x128_epilogue.sm,@object
    .local  llvm.mtgpu.kernel.asm_bf16bf16gemm_nn_tc_128_32x128_epilogue.sm
    .comm   llvm.mtgpu.kernel.asm_bf16bf16gemm_nn_tc_128_32x128_epilogue.sm,10240,4

    .addrsig
    .mtgpu_metadata.start
---
mtgpu.kernels:
  - .arch_id:        220
    .args:
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           output_ptr
          .offset:         0
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           out_buffer_size
          .offset:         2
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           _0
          .offset:         3
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           input_a
          .offset:         4
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           a_buffer_size
          .offset:         6
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           _1
          .offset:         7
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           input_b
          .offset:         8
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           b_buffer_size
          .offset:         10
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           _2
          .offset:         11
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           input_bias
          .offset:         12
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           bias_buffer_size
          .offset:         14
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           _3
          .offset:         15
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           input_z
          .offset:         16
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           input_z_buffer_size
          .offset:         18
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           _4
          .offset:         19
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           input_a_scales
          .offset:         20
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           a_scales_buffer_size
          .offset:         22
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           _5
          .offset:         23
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           input_b_scales
          .offset:         24
          .reg_bank:       3
          .size:           8
          .type:           14
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           b_scales_buffer_size
          .offset:         26
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           _6
          .offset:         27
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           nr_m
          .offset:         28
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           nr_n
          .offset:         29
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           nr_k
          .offset:         30
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           lda
          .offset:         31
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           ldb
          .offset:         32
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           ldc
          .offset:         33
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           alpha
          .offset:         34
          .reg_bank:       3
          .size:           4
          .type:           8
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           beta
          .offset:         35
          .reg_bank:       3
          .size:           4
          .type:           8
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           gamma
          .offset:         36
          .reg_bank:       3
          .size:           4
          .type:           8
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           nr_batch
          .offset:         37
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           batch_stride_a
          .offset:         38
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           batch_stride_b
          .offset:         39
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           batch_stride_c
          .offset:         40
          .reg_bank:       3
          .size:           4
          .type:           6
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           prog_size
          .offset:         41
          .reg_bank:       3
          .size:           4
          .type:           7
        .input_mode:     0
        .kind:           35
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           prog_addr
          .offset:         42
          .reg_bank:       3
          .size:           8
          .type:           10
        .input_mode:     0
        .kind:           33
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           gridDim_x
          .offset:         44
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           13
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           gridDim_y
          .offset:         45
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           14
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           gridDim_z
          .offset:         46
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           15
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockDim_x
          .offset:         47
          .reg_bank:       3
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           9
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           Global Spill Base
          .offset:         48
          .reg_bank:       3
          .size:           8
          .type:           18
        .input_mode:     0
        .kind:           19
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           Core Number
          .offset:         50
          .reg_bank:       3
          .size:           4
          .type:           7
        .input_mode:     0
        .kind:           23
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockIdx_x
          .offset:         4
          .reg_bank:       5
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           5
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockIdx_y
          .offset:         5
          .reg_bank:       5
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           6
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           blockIdx_z
          .offset:         6
          .reg_bank:       5
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           7
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           threadIdx_x
          .offset:         0
          .reg_bank:       2
          .size:           4
          .type:           0
        .input_mode:     0
        .kind:           1
    .async_bar_count: 0
    .attr_reg_count: 1
    .attrs:          {}
    .barrier_coef:   4294967295
    .clusterX:       0
    .clusterY:       0
    .clusterZ:       0
    .coef_reg_count: 0
    .const_calc_kernel: !str ''
    .constcalc_temp_reg_count: 0
    .global_atomic_count: 0
    .has_barrier:    false
    .has_cluster_barrier: false
    .has_texture: false
    .has_sqmma_conv:    false
    .imm_info:
      .offset:         51
      .reg_bank:       3
      .size:           56
      .values:
        - 9
        - 11
        - 32
        - 48
        - 63
        - 64
        - 80
        - 96
        - 112
        - 128
        - 160
        - 192
        - 224
        - 1061962282
    .indirect_call_flag: false
    .call_flag: false
    .internal_reg_count: 2
    .local_mem_used_in_cluster: 0
    .max_block_size: 1
    .name:           asm_bf16bf16gemm_nn_tc_128_32x128_epilogue
    .private_memory_size: 0
    .shared_memory_size: 10368
    .shared_reg_count: 65
    .slot_reg_count: 4
    .temp_reg_count: 155
    .wave_mode:      0
mtgpu.version:
  - 2
  - 2
...

    .mtgpu_metadata.end
