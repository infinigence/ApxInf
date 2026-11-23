
    .text
    .protected  asm_bf16bf16gemm_nn_tc_256_256x64_epilogue
    .globl  asm_bf16bf16gemm_nn_tc_256_256x64_epilogue
    .p2align    8
    .type   asm_bf16bf16gemm_nn_tc_256_256x64_epilogue,@function
asm_bf16bf16gemm_nn_tc_256_256x64_epilogue:







    ; init stage
[N--D---] FOP.MOV R32, C0
[N--D--s]   MOV.LD.LM.B32 R18, LM4
[N--D---] FOP.MOV R33, C0
[N--D--s]   MOV.LD.LM.B32 R19, LM5
[N--D---] FOP.MOV R34, C0
[N--D--o]   MOV.LD.LM.B32 R2, LM6
[N--D---] FOP.MOV R35, C0
[N-sD---]   BIT.ROT.LSL SL3, C79, R18
[N--D---] FOP.MOV R36, C0
[N--D---] FOP.MOV R37, C0
[NswD---]   BIT.ROT.LSL SL2, C46, R19
[N--D---] FOP.MOV R38, C0
[NwsD---]   BIT.ROT.LSR R0, C59, AR0
[N--D---] FOP.MOV R39, C0
[NswD---]   BIT.LUT.AND R1, AR0, SH66
[N--D---] FOP.MOV R40, C0
[NwsD---]   BIT.AND.LSL R20, C50, C0, AR0
[N--D---] FOP.MOV R41, C0
[NswD---]   BIT.ROT.LSL R21, C59, R0
[N--D---] FOP.MOV R42, C0
[NwsD---]   BIT.ROT.LSR R17, C0, R1
[N--D---] FOP.MOV R43, C0
[NswD---]       INT.ADD R21.s32, R17.s32, R21.s32
[N--D---] FOP.MOV R44, C0
[NwsD---]   BIT.ROT.LSL R22, C79, SH31
[N--D---] FOP.MOV R45, C0
[NswD---]   INT.ADD R16.s32, SH58.s32, C0.s32
[N--D---] FOP.MOV R46, C0
[NwsD---]   BIT.AND.LSL R23, C50, C59, AR0
[N--D---] FOP.MOV R47, C0
[N--D---] FOP.MOV R48, C0
[NswD---]   BIT.ROT.LSR R24, C50, AR0
[N--D---] FOP.MOV R49, C0
[NwsD---]   BIT.ROT.LSL R25, C80, SH32
[N--D---] FOP.MOV R50, C0
[NswD---]       INT.ADD R5.s32, SL2.s32, R21.s32
[N--D---] FOP.MOV R51, C0
[NwsD---]       INT.MADD R5.s32, SH31.s32, R20.s32, R5.s32
[N--D---] FOP.MOV R52, C0
[NswD---]       INT.MADD R5.s32, SH38.s32, R5.s32, R2.s32
[NwsD---]       BIT.ROT.LSL R5, C33, R5
[N--D---] FOP.MOV R53, C0
[NswD---]       INT.ADD R6.s32, R22.s32, R5.s32
[N--D---] FOP.MOV R54, C0
[NwsD---]       INT.ADD R7.s32, R22.s32, R6.s32
[N--D---] FOP.MOV R55, C0
[NswD---]       INT.ADD R8.s32, R22.s32, R7.s32
[N--D---] FOP.MOV R56, C0
[N--D---] FOP.MOV R57, C0
[NwsD---]       INT.MADD R9.s32, SH32.s32, R23.s32, R24.s32
[N--D---] FOP.MOV R58, C0
[NswD---]       INT.ADD  R9.s32, SL3.s32, R9.s32
[N--D---] FOP.MOV R59, C0
[NwsD---]       INT.MADD R9.s32, SH39.s32, R9.s32, R2.s32
[NswD---]       BIT.ROT.LSL R9, C33, R9
[NwsD---]       BIT.ROT.LSL R15, C79, SH32
[N--D---] FOP.MOV R60, C0
[NswD---]       FOP.MOV R3, R24
[NwsD---]       INT.BYP.TL p0, R3.s32, SH30.s32
[NswD---]       INT.BYP.MOVC.PRB R9.s32, R9.s32, SH10.s32, p0
[NwsD---]       INT.ADD R3.s32, R3.s32, SH53.s32
[N--D---] FOP.MOV R61, C0
[N--D---] FOP.MOV R62, C0
[NwwDs--]       DMA.LD.B512 R[122:137], SH[4:5], R5, _, _, _, _,  SH[4:6]
[N--D---] FOP.MOV R63, C0
[N--Ds--]       DMA.LD.B128 R[138:141], SH[8:9], R9, _, _, _, _,  SH[8:10]
[N--D---] FOP.MOV R64, C0
[N--D---] FOP.MOV R65, C0
[N--D---] FOP.MOV R66, C0
[N--D---] FOP.MOV R67, C0
[NswD---]       BIT.ROT.LSL R20, SH52, R0
[NwsD---]       BIT.ROT.LSR R23, C61, R1
[NswD---]       BIT.ROT.LSL R23, C46, R23
[NwsD---]       BIT.AND.LSL R26, C50, C111, R1
[NswD---]       INT.ADD R11.s32, R20.s32, R23.s32
[NwsD---]       INT.ADD R11.s32, R11.s32, R26.s32
[NswD---]       BIT.ROT.LSR R11, C33, R11
[N--D---] FOP.MOV R68, C0
[N--D---] FOP.MOV R69, C0
[N--D---] FOP.MOV R70, C0
[N--D---] FOP.MOV R71, C0
[N--D---] FOP.MOV R72, C0
[N--D---] FOP.MOV R73, C0
[N--D---] FOP.MOV R74, C0
[N--D---] FOP.MOV R75, C0
[N--D---] FOP.MOV R76, C0
[N--D---] FOP.MOV R77, C0
[N--D---] FOP.MOV R78, C0
[N--D---] FOP.MOV R79, C0
[NwsD---]       BIT.ROT.LSL R26, SH51, R0
[NswD---]       BIT.AND.LSR R28, C33, C59, R1
[NwsD---]       BIT.ROT.LSL R28, C46, R28
[NswD---]       BIT.ROT.LSR R30, C80, R1
[NwsD---]       BIT.ROT.LSL R30, C79, R30
[NswD---]       BIT.AND.LSR R31, C50, C78, R1
[NwsD---]       BIT.ROT.LSL R31, C61, R31
[NswD---]       BIT.AND.LSL R17, C50, C33, R1
[NwsD---]       INT.ADD R31.s32, R17.s32, R31.s32
[NswD---]       INT.ADD R12.s32, R26.s32, R28.s32
[NwsD---]       INT.ADD R12.s32, R12.s32, R30.s32
[NswD---]       INT.ADD R12.s32, R12.s32, R31.s32
[NwsD---]       BIT.ROT.LSR R12, C33, R12
[N--D---] FOP.MOV R80, C0
[N--D---] FOP.MOV R81, C0
[NswD---]       BIT.ROT.LSL R26, SH52, R0
[N--D---] FOP.MOV R82, C0
[NwsD---]       INT.MADD R27.s32, C45.s32, R26.s32, R1.s32
[N--D---] FOP.MOV R83, C0
[NswD---]       BIT.ROT.LSR R13, C33, R27
[N--D---] FOP.MOV R84, C0
[NwsD---]       BIT.ROT.LSL R14, C0, R1
[N--D---] FOP.MOV R85, C0
[NswD---]       INT.ADD SL1.s32, SH30.s32, SH53.neg.s32
[N--D---] FOP.MOV R86, C0
[N--D---] FOP.MOV R87, C0
[N--D---] FOP.MOV R88, C0
[N--D---] FOP.MOV R89, C0
[N--D---] FOP.MOV R90, C0
[N--D---] FOP.MOV R91, C0
[N--D---] FOP.MOV R92, C0
[N--Dw--] FOP.MOV R93, C0
[N--D---] FOP.MOV R94, C0
[N-sD---] FOP.MOV R95, C0



[N-wD-s-]       LDMA.WR.B128.BST1 R[122:125], R11, X4, 128 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[126:129], R11, X4, 384 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[130:133], R11, X4, 4224 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[134:137], R11, X4, 4480 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[138:141], R12, X4, 16512 , BURST_STRIDE=16
[N-wD---]       INT.ADD R5.s32, R16.s32, R5.s32
[N--D---]       INT.ADD R6.s32, R16.s32, R6.s32
[N--D---]       INT.ADD R7.s32, R16.s32, R7.s32
[N--D---]       INT.ADD R8.s32, R16.s32, R8.s32
[N-sD-w-]       INT.ADD R9.s32, R15.s32, R9.s32
[NswD---]       INT.BYP.TL p0, R3.s32, SH30.s32
[NwsD---]       INT.BYP.MOVC.PRB R9.s32, R9.s32, SH10.s32, p0
[NswD---]       INT.ADD R3.s32, R3.s32, SH53.s32
[NwwD---]       AP.BARRIER LM0
[N-wD-s-]       LDMA.RD.B32.BST4  R[96:99], R13, X4, 128, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[100:103], R13, X4, 2176, BURST_STRIDE=512
[N--D-o-]       LDMA.RD.B32.BST4  R[113:116], R14, X4, 16512, BURST_STRIDE=512
[NswD--w]       INT.BYP.TG p0, SL1.s32, C0.s32
[Nw-D---]       CTRL.BR.ALLP0F LAST_LOOP


MAIN_LOOP:

[N--D---] MMA.323216 R32[4:2][2:1].f32, R96[2:1][1:1].bf16.row, R113[1:1][2:1].bf16.row, R32[4:2][2:1].f32
[NwwDs--]       DMA.LD.B512 R[122:137], SH[4:5], R5, _, _, _, _,  SH[4:6]
[N--D---] MMA.323216 R40[4:2][2:1].f32, R96[2:1][1:1].bf16.row, R115[1:1][2:1].bf16.row, R40[4:2][2:1].f32
[N--Ds--]       DMA.LD.B128 R[138:141], SH[8:9], R9, _, _, _, _,  SH[8:10]
[N--D---] MMA.323216 R48[4:2][2:1].f32, R98[2:1][1:1].bf16.row, R113[1:1][2:1].bf16.row, R48[4:2][2:1].f32
[N--D---] MMA.323216 R56[4:2][2:1].f32, R98[2:1][1:1].bf16.row, R115[1:1][2:1].bf16.row, R56[4:2][2:1].f32
[N--D---] MMA.323216 R64[4:2][2:1].f32, R100[2:1][1:1].bf16.row, R113[1:1][2:1].bf16.row, R64[4:2][2:1].f32
[N--D---] MMA.323216 R72[4:2][2:1].f32, R100[2:1][1:1].bf16.row, R115[1:1][2:1].bf16.row, R72[4:2][2:1].f32
[N-wD--s]       LDMA.RD.B32.BST4  R[104:107], R13, X4, 4224, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[108:111], R13, X4, 6272, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[117:120], R14, X4, 18560, BURST_STRIDE=512
[N--D---] MMA.323216 R80[4:2][2:1].f32, R102[2:1][1:1].bf16.row, R113[1:1][2:1].bf16.row, R80[4:2][2:1].f32
[Ns-D--w] MMA.323216 R88[4:2][2:1].f32, R102[2:1][1:1].bf16.row, R115[1:1][2:1].bf16.row, R88[4:2][2:1].f32
[N--D---] MMA.323216 R32[4:2][2:1].f32, R104[2:1][1:1].bf16.row, R117[1:1][2:1].bf16.row, R32[4:2][2:1].f32
[NwwD---]   AP.BARRIER LM0
[N--Dw--]   CTRL.NOP
[N--D---] MMA.323216 R40[4:2][2:1].f32, R104[2:1][1:1].bf16.row, R119[1:1][2:1].bf16.row, R40[4:2][2:1].f32
[N-wD-s-]       LDMA.WR.B128.BST1 R[122:125], R11, X4, 128 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[126:129], R11, X4, 384 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[130:133], R11, X4, 4224 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[134:137], R11, X4, 4480 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST1 R[138:141], R12, X4, 16512 , BURST_STRIDE=16
[N--D---] MMA.323216 R48[4:2][2:1].f32, R106[2:1][1:1].bf16.row, R117[1:1][2:1].bf16.row, R48[4:2][2:1].f32
[N--D-w-]   CTRL.NOP
[NwwD---]   AP.BARRIER LM0
[N--D---] MMA.323216 R56[4:2][2:1].f32, R106[2:1][1:1].bf16.row, R119[1:1][2:1].bf16.row, R56[4:2][2:1].f32
[N--D---]       INT.ADD R5.s32, R16.s32, R5.s32
[N--D---]       INT.ADD R6.s32, R16.s32, R6.s32
[N--D---]       INT.ADD R7.s32, R16.s32, R7.s32
[N--D---]       INT.ADD R8.s32, R16.s32, R8.s32
[N-sD---]       INT.ADD R9.s32, R15.s32, R9.s32
[N--D---] MMA.323216 R64[4:2][2:1].f32, R108[2:1][1:1].bf16.row, R117[1:1][2:1].bf16.row, R64[4:2][2:1].f32
[NswD---]       INT.BYP.TL p0, R3.s32, SH30.s32
[NwsD---]       INT.BYP.MOVC.PRB R9.s32, R9.s32, SH10.s32, p0
[NswD---]       INT.ADD R3.s32, R3.s32, SH53.s32
[N--D---] MMA.323216 R72[4:2][2:1].f32, R108[2:1][1:1].bf16.row, R119[1:1][2:1].bf16.row, R72[4:2][2:1].f32
[Nw-D--s]       LDMA.RD.B32.BST4  R[96:99], R13, X4, 128, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[100:103], R13, X4, 2176, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[113:116], R14, X4, 16512, BURST_STRIDE=512
[N--D---] MMA.323216 R80[4:2][2:1].f32, R110[2:1][1:1].bf16.row, R117[1:1][2:1].bf16.row, R80[4:2][2:1].f32
[N-sD--w] MMA.323216 R88[4:2][2:1].f32, R110[2:1][1:1].bf16.row, R119[1:1][2:1].bf16.row, R88[4:2][2:1].f32
[Ns-D---]    INT.ADD SL1.s32, SL1.s32, SH53.neg.s32
[NwsD---]    INT.BYP.TG p0, SL1.s32, C0.s32
[N-wD---]    CTRL.BR.ANYP0T MAIN_LOOP


LAST_LOOP:
[N-wD---] MMA.323216 R32[4:2][2:1].f32, R96[2:1][1:1].bf16.row, R113[1:1][2:1].bf16.row, R32[4:2][2:1].f32
[NswD---]       BIT.ROT.LSR R26, C79, R1
[NwsD---]       BIT.ROT.LSL R26, C45, R26
[NswD---]       BIT.AND.LSR R3, C50, C78, R1
[N--D---] MMA.323216 R40[4:2][2:1].f32, R96[2:1][1:1].bf16.row, R115[1:1][2:1].bf16.row, R40[4:2][2:1].f32
[NwsD---]       INT.ADD R26.s32, R3.s32, R26.s32
[NswD---]       INT.MADD R26.s32, SH67.s32, R26.s32, R0.s32
[NwsD---]       INT.ADD R26.s32, SL2.s32, R26.s32
[NswD---]       BIT.AND.LSR R27, C80, SH57, R1
[N--D---] MMA.323216 R48[4:2][2:1].f32, R98[2:1][1:1].bf16.row, R113[1:1][2:1].bf16.row, R48[4:2][2:1].f32
[NwsD---]       BIT.ROT.LSL R27, C50, R27
[NswD---]       BIT.LUT.AND R3, C59, R1
[N--D---] MMA.323216 R56[4:2][2:1].f32, R98[2:1][1:1].bf16.row, R115[1:1][2:1].bf16.row, R56[4:2][2:1].f32
[NwsD---]       INT.ADD R27.s32, R3.s32, R27.s32
[NswD---]       INT.ADD R27.s32, SL3.s32, R27.s32
[N--D---] MMA.323216 R64[4:2][2:1].f32, R100[2:1][1:1].bf16.row, R113[1:1][2:1].bf16.row, R64[4:2][2:1].f32
[N--D---] MMA.323216 R72[4:2][2:1].f32, R100[2:1][1:1].bf16.row, R115[1:1][2:1].bf16.row, R72[4:2][2:1].f32
[N-wD--s]       LDMA.RD.B32.BST4  R[104:107], R13, X4, 4224, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[108:111], R13, X4, 6272, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[117:120], R14, X4, 18560, BURST_STRIDE=512
[N--D---] MMA.323216 R80[4:2][2:1].f32, R102[2:1][1:1].bf16.row, R113[1:1][2:1].bf16.row, R80[4:2][2:1].f32
[Ns-D--w] MMA.323216 R88[4:2][2:1].f32, R102[2:1][1:1].bf16.row, R115[1:1][2:1].bf16.row, R88[4:2][2:1].f32
[N-wD---] MMA.323216 R32[4:2][2:1].f32, R104[2:1][1:1].bf16.row, R117[1:1][2:1].bf16.row, R32[4:2][2:1].f32
[NswD---]       FOP.MOV R150, C0
[N--D---]       FOP.MOV R151, C0
[N--D---]       FOP.MOV R152, C0
[NwsD---]       FOP.MOV R153, C0
[N--D---] MMA.323216 R40[4:2][2:1].f32, R104[2:1][1:1].bf16.row, R119[1:1][2:1].bf16.row, R40[4:2][2:1].f32
[N--D---] MMA.323216 R48[4:2][2:1].f32, R106[2:1][1:1].bf16.row, R117[1:1][2:1].bf16.row, R48[4:2][2:1].f32
[NwsD---]        BIT.LUT.OR.TZ p0, SH13, SH12
[NswD---]        INT.BYP.TE.OP p0, SH36.s32, C0.s32
[NwwD---]        CTRL.BR.ANYP0T BB0_BIAS_END
[NwsD---]    BIT.ROT.LSL R150, C33, R27
[N-wD---]    INT.ADD R151.s32, SH53.s32, R150.s32
[N--D---]    INT.ADD R152.s32, SH58.s32, R150.s32
[N-sD---]    INT.ADD R153.s32, SH62.s32, R150.s32
[N-wD-s-]        DMA.LD.B16 R150.l, SH[12:13], R150, _, _, _, _,  SH[12:14]
[N--D-s-]        DMA.LD.B16 R151.l, SH[12:13], R151, _, _, _, _,  SH[12:14]
[N--D-s-]        DMA.LD.B16 R152.l, SH[12:13], R152, _, _, _, _,  SH[12:14]
[N--D-s-]        DMA.LD.B16 R153.l, SH[12:13], R153, _, _, _, _,  SH[12:14]
BB0_BIAS_END:
[N--D---] MMA.323216 R56[4:2][2:1].f32, R106[2:1][1:1].bf16.row, R119[1:1][2:1].bf16.row, R56[4:2][2:1].f32
[N--D---] MMA.323216 R64[4:2][2:1].f32, R108[2:1][1:1].bf16.row, R117[1:1][2:1].bf16.row, R64[4:2][2:1].f32
[N--D---] MMA.323216 R72[4:2][2:1].f32, R108[2:1][1:1].bf16.row, R119[1:1][2:1].bf16.row, R72[4:2][2:1].f32
[NwsD---]       INT.MADD R28.s32, SH33.s32, R27.s32, R26.s32
[NswD---]       INT.MADD R28.s32, SH40.s32, R28.s32, R2.s32
[NwsD---]       BIT.ROT.LSL R28, C33, R28
[NswD---]       FOP.MOV R30, SH33
[NwsD---]       INT.ADD R3.s32, SH28.s32, R26.neg.s32
[NswD---]       INT.ADD R4.s32, SH29.s32, R27.neg.s32
[N--D---] MMA.323216 R80[4:2][2:1].f32, R110[2:1][1:1].bf16.row, R117[1:1][2:1].bf16.row, R80[4:2][2:1].f32
[NwsD---]       INT.ADD R128.s32, C0.s32, R28.s32
[NswD---]       INT.ADD R129.s32, SH53.s32, R28.s32
[NwsD---]       INT.ADD R130.s32, SH58.s32, R28.s32
[NswD---]       INT.ADD R131.s32, SH62.s32, R28.s32
[NwsD---]       INT.MUL R29.s32, C39.s32,  R30.s32
[N-sDww-] MMA.323216 R88[4:2][2:1].f32, R110[2:1][1:1].bf16.row, R119[1:1][2:1].bf16.row, R88[4:2][2:1].f32
; core loop stage end

; debug use



[NswD---]       INT.BYP.TLE p0, R3.s32, SH65.s32
[NwsD---]       INT.BYP.TLE.OP p0, R4.s32, SH55.s32
[NwwD---]        CTRL.BR.ANYP0T STORE_MASK
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R32.l.bf16, SH34, R32, R96
[N--D---] FOP.FMA R33.l.bf16, SH34, R33, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R40.l.bf16, SH34, R40, R98
[N-sD---] FOP.FMA R41.l.bf16, SH34, R41, R99
[N-wD---]       DMA.ST.B16 R32.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R33.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R40.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R41.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R34.l.bf16, SH34, R34, R96
[N--D---] FOP.FMA R35.l.bf16, SH34, R35, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R42.l.bf16, SH34, R42, R98
[N-sD---] FOP.FMA R43.l.bf16, SH34, R43, R99
[N-wD---]       DMA.ST.B16 R34.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R35.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R42.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R43.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R36.l.bf16, SH34, R36, R96
[N--D---] FOP.FMA R37.l.bf16, SH34, R37, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R44.l.bf16, SH34, R44, R98
[N-sD---] FOP.FMA R45.l.bf16, SH34, R45, R99
[N-wD---]       DMA.ST.B16 R36.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R37.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R44.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R45.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R38.l.bf16, SH34, R38, R96
[N--D---] FOP.FMA R39.l.bf16, SH34, R39, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R46.l.bf16, SH34, R46, R98
[N-sD---] FOP.FMA R47.l.bf16, SH34, R47, R99
[N-wD---]       DMA.ST.B16 R38.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R39.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R46.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R47.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R48.l.bf16, SH34, R48, R96
[N--D---] FOP.FMA R49.l.bf16, SH34, R49, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R56.l.bf16, SH34, R56, R98
[N-sD---] FOP.FMA R57.l.bf16, SH34, R57, R99
[N-wD---]       DMA.ST.B16 R48.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R49.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R56.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R57.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R50.l.bf16, SH34, R50, R96
[N--D---] FOP.FMA R51.l.bf16, SH34, R51, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R58.l.bf16, SH34, R58, R98
[N-sD---] FOP.FMA R59.l.bf16, SH34, R59, R99
[N-wD---]       DMA.ST.B16 R50.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R51.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R58.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R59.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R52.l.bf16, SH34, R52, R96
[N--D---] FOP.FMA R53.l.bf16, SH34, R53, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R60.l.bf16, SH34, R60, R98
[N-sD---] FOP.FMA R61.l.bf16, SH34, R61, R99
[N-wD---]       DMA.ST.B16 R52.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R53.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R60.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R61.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R54.l.bf16, SH34, R54, R96
[N--D---] FOP.FMA R55.l.bf16, SH34, R55, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R62.l.bf16, SH34, R62, R98
[N-sD---] FOP.FMA R63.l.bf16, SH34, R63, R99
[N-wD---]       DMA.ST.B16 R54.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R55.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R62.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R63.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R64.l.bf16, SH34, R64, R96
[N--D---] FOP.FMA R65.l.bf16, SH34, R65, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R72.l.bf16, SH34, R72, R98
[N-sD---] FOP.FMA R73.l.bf16, SH34, R73, R99
[N-wD---]       DMA.ST.B16 R64.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R65.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R72.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R73.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R66.l.bf16, SH34, R66, R96
[N--D---] FOP.FMA R67.l.bf16, SH34, R67, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R74.l.bf16, SH34, R74, R98
[N-sD---] FOP.FMA R75.l.bf16, SH34, R75, R99
[N-wD---]       DMA.ST.B16 R66.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R67.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R74.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R75.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R68.l.bf16, SH34, R68, R96
[N--D---] FOP.FMA R69.l.bf16, SH34, R69, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R76.l.bf16, SH34, R76, R98
[N-sD---] FOP.FMA R77.l.bf16, SH34, R77, R99
[N-wD---]       DMA.ST.B16 R68.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R69.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R76.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R77.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R70.l.bf16, SH34, R70, R96
[N--D---] FOP.FMA R71.l.bf16, SH34, R71, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R78.l.bf16, SH34, R78, R98
[N-sD---] FOP.FMA R79.l.bf16, SH34, R79, R99
[N-wD---]       DMA.ST.B16 R70.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R71.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R78.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R79.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R80.l.bf16, SH34, R80, R96
[N--D---] FOP.FMA R81.l.bf16, SH34, R81, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R88.l.bf16, SH34, R88, R98
[N-sD---] FOP.FMA R89.l.bf16, SH34, R89, R99
[N-wD---]       DMA.ST.B16 R80.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R81.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R88.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R89.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R82.l.bf16, SH34, R82, R96
[N--D---] FOP.FMA R83.l.bf16, SH34, R83, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R90.l.bf16, SH34, R90, R98
[N-sD---] FOP.FMA R91.l.bf16, SH34, R91, R99
[N-wD---]       DMA.ST.B16 R82.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R83.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R90.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R91.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R84.l.bf16, SH34, R84, R96
[N--D---] FOP.FMA R85.l.bf16, SH34, R85, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R92.l.bf16, SH34, R92, R98
[N-sD---] FOP.FMA R93.l.bf16, SH34, R93, R99
[N-wD---]       DMA.ST.B16 R84.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R85.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R92.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R93.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R86.l.bf16, SH34, R86, R96
[N--D---] FOP.FMA R87.l.bf16, SH34, R87, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N--D---] FOP.FMA R94.l.bf16, SH34, R94, R98
[N-sD---] FOP.FMA R95.l.bf16, SH34, R95, R99
[N-wD---]       DMA.ST.B16 R86.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R87.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[N--D---]       DMA.ST.B16 R94.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[N-sD---]       DMA.ST.B16 R95.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[N-wD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
  ; [NwsD---]       FOP.MUL R17, SH68, R17
  ; [NwsD---]       FOP.MUL R17, C81, R17
  ; [NswD---]       BIT.ROT.LSL R31, C45, AR0
  ; [NwsD---] MOV.B32  R11, SH0[R10]
  ; [NwsD---]       DMA.ST.B32 R5, SH[0:1], R31
  [NwwD---]   CTRL.NOP
  [NwwD---]   AP.BARRIER LM0
  [NwwD---]   CTRL.END

STORE_MASK:
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R32.l.bf16, SH34, R32, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R33.l.bf16, SH34, R33, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R40.l.bf16, SH34, R40, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R41.l.bf16, SH34, R41, R99
[NswD---]       INT.BYP.TG p0, R3.s32, C0.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R32.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R33.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R40.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NswD---]   P0  DMA.ST.B16 R41.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R34.l.bf16, SH34, R34, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R35.l.bf16, SH34, R35, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R42.l.bf16, SH34, R42, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R43.l.bf16, SH34, R43, R99
[NwsD---]       INT.BYP.TG p0, R3.s32, C46.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R34.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R35.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R42.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NwsD---]   P0  DMA.ST.B16 R43.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R36.l.bf16, SH34, R36, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R37.l.bf16, SH34, R37, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R44.l.bf16, SH34, R44, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R45.l.bf16, SH34, R45, R99
[NswD---]       INT.BYP.TG p0, R3.s32, C39.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R36.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R37.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R44.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NswD---]   P0  DMA.ST.B16 R45.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R38.l.bf16, SH34, R38, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R39.l.bf16, SH34, R39, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R46.l.bf16, SH34, R46, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R47.l.bf16, SH34, R47, R99
[NwsD---]       INT.BYP.TG p0, R3.s32, C48.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R38.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R39.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R46.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NwsD---]   P0  DMA.ST.B16 R47.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R48.l.bf16, SH34, R48, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R49.l.bf16, SH34, R49, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R56.l.bf16, SH34, R56, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R57.l.bf16, SH34, R57, R99
[NswD---]       INT.BYP.TG p0, R3.s32, SH53.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R48.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R49.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R56.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NswD---]   P0  DMA.ST.B16 R57.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R50.l.bf16, SH34, R50, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R51.l.bf16, SH34, R51, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R58.l.bf16, SH34, R58, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R59.l.bf16, SH34, R59, R99
[NwsD---]       INT.BYP.TG p0, R3.s32, SH54.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R50.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R51.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R58.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NwsD---]   P0  DMA.ST.B16 R59.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R52.l.bf16, SH34, R52, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R53.l.bf16, SH34, R53, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R60.l.bf16, SH34, R60, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R61.l.bf16, SH34, R61, R99
[NswD---]       INT.BYP.TG p0, R3.s32, SH55.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R52.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R53.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R60.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NswD---]   P0  DMA.ST.B16 R61.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R54.l.bf16, SH34, R54, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R55.l.bf16, SH34, R55, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R62.l.bf16, SH34, R62, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R63.l.bf16, SH34, R63, R99
[NwsD---]       INT.BYP.TG p0, R3.s32, SH56.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R54.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R55.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R62.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NwsD---]   P0  DMA.ST.B16 R63.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R64.l.bf16, SH34, R64, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R65.l.bf16, SH34, R65, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R72.l.bf16, SH34, R72, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R73.l.bf16, SH34, R73, R99
[NswD---]       INT.BYP.TG p0, R3.s32, SH58.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R64.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R65.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R72.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NswD---]   P0  DMA.ST.B16 R73.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R66.l.bf16, SH34, R66, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R67.l.bf16, SH34, R67, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R74.l.bf16, SH34, R74, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R75.l.bf16, SH34, R75, R99
[NwsD---]       INT.BYP.TG p0, R3.s32, SH59.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R66.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R67.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R74.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NwsD---]   P0  DMA.ST.B16 R75.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R68.l.bf16, SH34, R68, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R69.l.bf16, SH34, R69, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R76.l.bf16, SH34, R76, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R77.l.bf16, SH34, R77, R99
[NswD---]       INT.BYP.TG p0, R3.s32, SH60.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R68.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R69.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R76.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NswD---]   P0  DMA.ST.B16 R77.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R70.l.bf16, SH34, R70, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R71.l.bf16, SH34, R71, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R78.l.bf16, SH34, R78, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R79.l.bf16, SH34, R79, R99
[NwsD---]       INT.BYP.TG p0, R3.s32, SH61.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R70.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R71.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R78.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NwsD---]   P0  DMA.ST.B16 R79.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R80.l.bf16, SH34, R80, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R81.l.bf16, SH34, R81, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R88.l.bf16, SH34, R88, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R89.l.bf16, SH34, R89, R99
[NswD---]       INT.BYP.TG p0, R3.s32, SH62.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R80.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R81.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R88.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NswD---]   P0  DMA.ST.B16 R89.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R82.l.bf16, SH34, R82, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R83.l.bf16, SH34, R83, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R90.l.bf16, SH34, R90, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R91.l.bf16, SH34, R91, R99
[NwsD---]       INT.BYP.TG p0, R3.s32, SH63.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R82.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R83.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R90.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NwsD---]   P0  DMA.ST.B16 R91.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R84.l.bf16, SH34, R84, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R85.l.bf16, SH34, R85, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R92.l.bf16, SH34, R92, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R93.l.bf16, SH34, R93, R99
[NswD---]       INT.BYP.TG p0, R3.s32, SH64.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R84.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NswD---]   P0  DMA.ST.B16 R85.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NswD---]   P0  DMA.ST.B16 R92.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NswD---]   P0  DMA.ST.B16 R93.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32
[N-wD---] FOP.MUL R96, R150.l.bf16, SH36
[N--D---] FOP.FMA R86.l.bf16, SH34, R86, R96
[N--D---] FOP.MUL R97, R151.l.bf16, SH36
[N--D---] FOP.FMA R87.l.bf16, SH34, R87, R97
[N--D---] FOP.MUL R98, R152.l.bf16, SH36
[N--D---] FOP.FMA R94.l.bf16, SH34, R94, R98
[N--D---] FOP.MUL R99, R153.l.bf16, SH36
[N-sD---] FOP.FMA R95.l.bf16, SH34, R95, R99
[NwsD---]       INT.BYP.TG p0, R3.s32, SH65.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R86.l, SH[0:1], R128, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C39.s32
[NwsD---]   P0  DMA.ST.B16 R87.l, SH[0:1], R129, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH53.s32
[NwsD---]   P0  DMA.ST.B16 R94.l, SH[0:1], R130, _, _, _, _,  SH[0:2]
[NswD---]       INT.BYP.TG.AP p0, R4.s32, SH55.s32
[NwsD---]   P0  DMA.ST.B16 R95.l, SH[0:1], R131, _, _, _, _,  SH[0:2]
[NwwD---] INT.ADD R128.s32, R29.s32, R128.s32
[N--D---] INT.ADD R129.s32, R29.s32, R129.s32
[N--D---] INT.ADD R130.s32, R29.s32, R130.s32
[N-sD---] INT.ADD R131.s32, R29.s32, R131.s32


[NwwD---]   AP.BARRIER LM0
[NwwD---]   CTRL.END

.Lfunc_end0:
    .size   asm_bf16bf16gemm_nn_tc_256_256x64_epilogue, .Lfunc_end0-asm_bf16bf16gemm_nn_tc_256_256x64_epilogue




    .type   llvm.mtgpu.kernel.asm_bf16bf16gemm_nn_tc_256_256x64_epilogue.sm,@object
    .local  llvm.mtgpu.kernel.asm_bf16bf16gemm_nn_tc_256_256x64_epilogue.sm
    .comm   llvm.mtgpu.kernel.asm_bf16bf16gemm_nn_tc_256_256x64_epilogue.sm,20480,4

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
    .barrier_coef:   0
    .clusterX:       0
    .clusterY:       0
    .clusterZ:       0
    .coef_reg_count: 0
    .const_calc_kernel: !str ''
    .constcalc_temp_reg_count: 0
    .global_atomic_count: 0
    .has_barrier:    true
    .has_cluster_barrier: false
    .has_texture: false
    .has_sqmma_conv:    false
    .imm_info:
      .offset:         51
      .reg_bank:       3
      .size:           72
      .values:
        - 10
        - 12
        - 32
        - 40
        - 48
        - 56
        - 63
        - 64
        - 72
        - 80
        - 88
        - 96
        - 104
        - 112
        - 120
        - 127
        - 128
        - 1061962282
    .indirect_call_flag: false
    .call_flag: false
    .internal_reg_count: 2
    .local_mem_used_in_cluster: 0
    .max_block_size: 1
    .name:           asm_bf16bf16gemm_nn_tc_256_256x64_epilogue
    .private_memory_size: 0
    .shared_memory_size: 20608
    .shared_reg_count: 69
    .slot_reg_count: 4
    .temp_reg_count: 169
    .wave_mode:      0
mtgpu.version:
  - 2
  - 2
...

    .mtgpu_metadata.end
