
    .text
    
    .p2align    8
    .type reset_ic_idx,@function
    reset_ic_idx:
        
[Ns-D---]    FOP.MOV R10, C0
[N-sD---]    INT.ADD SL1.s32, C33.s32, SL1.s32
[N--D---]    INT.ADD R3.s32, SH34.s32, R3.s32
[N--D---]    INT.ADD R5.s32, SH49.s32, R5.s32
[Nw-D---]    INT.ADD R7.s32, SH54.s32, R7.s32
[NswD---]    INT.BYP.TL p0 SL1.s32, SH25.s32
[NwwD---]    CTRL.BR.ANYP0T BB0_s_end
[N--D---]    INT.ADD R3.s32, SH45.neg.s32, R3.s32
[N--D---]    INT.ADD R5.s32, SH50.s32, R5.s32
[Ns-D---]    INT.ADD R4.s32, SH33.s32, R4.s32
[NwsD---]    FOP.MOV SL1, C0
BB0_s_end:
[NswD---]    INT.BYP.TL p0, R4.u32, SH22.u32
[NwsD---]    INT.BYP.TL.AP p0, R3.u32, SH23.u32
[NswD---]    INT.BYP.MOVC.PRB R6.s32, R5.s32, SH6.s32, p0
[NwwD---]    CTRL.BASR SL0


    .Lfunc_end0:
    .size	reset_ic_idx, .Lfunc_end0-reset_ic_idx
        
	.text
	.protected	musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16
	.globl	musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16
	.p2align	8
	.type	musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16,@function
musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16:








    ; init stage
[N--D--o]   MOV.LD.LM.B32 R18, LM4
[N--D-o-]   MOV.LD.LM.B32 R19, LM5
[N--Do--]   MOV.LD.LM.B32 R2, LM6
[NswD---]   FOP.MOV SL1, C0
[NwsD---]   BIT.ROT.LSL SL3, C79, R18
[NswD---]   BIT.ROT.LSL SL2, C46, R19
[NwsD---]   BIT.ROT.LSR R0, C59, AR0
[NswD---]   BIT.LUT.AND R1, AR0, SH103
[NwsD---]   BIT.AND.LSL R20, C61, C0, AR0
[NswD---]   BIT.ROT.LSL R21, C59, R0
[NwsD---]   BIT.ROT.LSR R17, C0, R1
[NswD---]       INT.ADD R21.s32, R17.s32, R21.s32
[NwsD---]   BIT.AND.LSL R23, C61, C50, AR0
[NswD---]   BIT.ROT.LSR R9, C45, AR0
[NwsD---]    FOP.MOV R10, C0
[NswD---]   INT.ADD R11.s32, SH21.s32, C0.s32
[NwsD---]   INT.ADD R25.s32, SL2.s32, R21.s32
[NswD---]   INT.BYP.TE p0, SH35.s32, C33.s32
[NwsD---] !p0 INT.MUL R[30:31].u64, SH36.u32, R25.u32
[NswD---] !p0 BIT.ROT.LSR R26, SH37, R31
[NwsD---]   INT.BYP.MOVC.PRB R26.s32, R25.s32, R26.s32, p0
[NswD---]   INT.MADD R27.s32, SH35.neg.s32, R25.s32, R26.s32
[NwsD---]   INT.BYP.TE p0, SH38.s32, C33.s32
[NswD---] !p0 INT.MUL R[30:31].u64, SH39.u32, R27.u32
[NwsD---] !p0 BIT.ROT.LSR R28, SH40, R31
[NswD---]   INT.BYP.MOVC.PRB R28.s32, R27.s32, R28.s32, p0
[NwsD---]   INT.MADD R29.s32, SH38.neg.s32, R27.s32, R28.s32
[NswD---]   INT.MADD R4.s32, SH29.s32, SH31.neg.s32, R28.s32
[NwsD---]   INT.MADD R3.s32, SH30.s32, SH32.neg.s32, R29.s32
[NswD---]   INT.BYP.TL p0, R4.u32, SH22.u32
[NwsD---]   INT.BYP.TL.AP p0, R3.u32, SH23.u32
[NswD---]     INT.MADD R5.s32, SH22.s32, R4.s32, R26.s32
[NwsD---]     INT.MADD R5.s32, SH23.s32, R3.s32, R5.s32
[NswD---]     INT.MADD R5.s32, SH59.s32, R2.s32, R5.s32
[NwsD---]     INT.MADD R5.s32, SH21.s32, R20.s32, R5.s32
[NswD---]     BIT.ROT.LSL R5, C0, R5
[NwsD---]   INT.BYP.MOVC.PRB R6.s32, R5.s32, SH6.s32, p0
[NswD---]       INT.ADD R7.s32, SL3.s32, R9.s32
[NwsD---]       INT.MADD R7.s32, SH51.s32, R7.s32, R2.s32
[NswD---]       INT.MADD R7.s32, SH46.s32, R23.s32, R7.s32
[NwsD---]       BIT.ROT.LSL R7, C0, R7
[NswD---]       BIT.ROT.LSR R8, C79, R1
[NwsD---]       BIT.ROT.LSL R8, C61, R8
[NswD---]       BIT.AND.LSL R17, C45, C50, R1
[NwsD---]       INT.ADD R8.s32, R17.s32, R8.s32
[NswD---]       INT.ADD R17.s32, C33.s32, R8.s32
[NwsD---]       INT.ADD R20.s32, C45.s32, R8.s32
[NswD---]       INT.ADD R21.s32, C50.s32, R8.s32
[NwsD---]       BIT.LUT.AND R28, SH94, SH21
[NswD---]       INT.BYP.TE p0, R28.s32, C0.s32
[NwsD---]       INT.BYP.MOVC.PRB R28.s32, SH95.s32, R28.s32, p0
[NswD---]       INT.BYP.TL p0, R8.s32, R28.s32
[NwsD---]       INT.BYP.MOVC.PRB R142.s32, C34.s32, C0.s32, p0
[NswD---]       INT.BYP.TL p0, R17.s32, R28.s32
[NwsD---]       INT.BYP.MOVC.PRB R142.s32, C52.s32, R142.s32, p0
[NswD---]       INT.BYP.TL p0, R20.s32, R28.s32
[NwsD---]       INT.BYP.MOVC.PRB R142.s32, SH112.s32, R142.s32, p0
[NswD---]       INT.BYP.TL p0, R21.s32, R28.s32
[NwsD---]       INT.BYP.MOVC.PRB R142.s32, C36.s32, R142.s32, p0
[NswD---]       INT.ADD R28.s32, R28.s32, SH90.neg.s32
[NwsD---]       INT.BYP.TL p0, R8.s32, R28.s32
[NswD---]       INT.BYP.MOVC.PRB R143.s32, C34.s32, C0.s32, p0
[NwsD---]       INT.BYP.TL p0, R17.s32, R28.s32
[NswD---]       INT.BYP.MOVC.PRB R143.s32, C52.s32, R143.s32, p0
[NwsD---]       INT.BYP.TL p0, R20.s32, R28.s32
[NswD---]       INT.BYP.MOVC.PRB R143.s32, SH112.s32, R143.s32, p0
[NwsD---]       INT.BYP.TL p0, R21.s32, R28.s32
[NswD---]       INT.BYP.MOVC.PRB R143.s32, C36.s32, R143.s32, p0
[NwsD---]       INT.BYP.TL p0, R10.s32, SH57.s32
[NswD---]       INT.BYP.MOVC.PRB R8.s32, C33.s32, C0.s32, p0
[NwwDs--]       DMA.LD.B512 R[122:137], SH[4:5], R6, _, _, _, _,  SH[4:6]
[N--Ds--]       DMA.LD.B128 R[138:141], SH[8:9], R7, _, _, _, _,  SH[8:10]
[NwsD---]       BIT.ROT.LSL R20, SH89, R0
[NswD---]       BIT.ROT.LSR R23, C61, R1
[NwsD---]       BIT.ROT.LSL R23, SH86, R23
[NswD---]       BIT.AND.LSL R26, C61, C111, R1
[NwsD---]       INT.ADD R13.s32, R20.s32, R23.s32
[NswD---]       INT.ADD R13.s32, R13.s32, R26.s32
[NwsD---]       BIT.ROT.LSR R13, C45, R13
[NswD---]       BIT.ROT.LSR R17, C45, R9
[NwsD---]       BIT.AND.LSL R9, C61, C50, R9
[NswD---]       INT.ADD R9.s32, R17.s32, R9.s32
[NwsD---]       BIT.AND.LSR R20, C33, C50, AR0
[NswD---]       BIT.ROT.LSL R20, SH88, R20
[NwsD---]       BIT.ROT.LSR R22, C61, R9
[NswD---]       BIT.ROT.LSL R22, SH86, R22
[NwsD---]       BIT.AND.LSL R23, C46, C33, AR0
[NswD---]       BIT.AND.LSR R25, C0, C111, R9
[NwsD---]       BIT.ROT.LSL R25, C61, R25
[NswD---]       INT.ADD R14.s32, R20.s32, R22.s32
[NwsD---]       INT.ADD R14.s32, R14.s32, R23.s32
[NswD---]       INT.ADD R14.s32, R14.s32, R25.s32
[NwsD---]       BIT.ROT.LSR R14, C45, R14
[NswD---]       BIT.ROT.LSL R26, SH89, R0
[NwsD---]       INT.MADD R27.s32, C61.s32, R26.s32, R1.s32
[NswD---]       BIT.ROT.LSR R15, C45, R27
[NwsD---]       BIT.ROT.LSL R16, C0, R1
[NswD---]       INT.ADD R12.s32, SH44.s32, SH95.neg.s32
[NwwD---] FOP.MOV R32, C0
[N--D---] FOP.MOV R33, C0
[N--D---] FOP.MOV R34, C0
[N--D---] FOP.MOV R35, C0
[N--D---] FOP.MOV R36, C0
[N--D---] FOP.MOV R37, C0
[N--D---] FOP.MOV R38, C0
[N--D---] FOP.MOV R39, C0
[N--D---] FOP.MOV R40, C0
[N--D---] FOP.MOV R41, C0
[N--D---] FOP.MOV R42, C0
[N--D---] FOP.MOV R43, C0
[N--D---] FOP.MOV R44, C0
[N--D---] FOP.MOV R45, C0
[N--D---] FOP.MOV R46, C0
[N--D---] FOP.MOV R47, C0
[N--D---] FOP.MOV R48, C0
[N--D---] FOP.MOV R49, C0
[N--D---] FOP.MOV R50, C0
[N--D---] FOP.MOV R51, C0
[N--D---] FOP.MOV R52, C0
[N--D---] FOP.MOV R53, C0
[N--D---] FOP.MOV R54, C0
[N--D---] FOP.MOV R55, C0
[N--D---] FOP.MOV R56, C0
[N--D---] FOP.MOV R57, C0
[N--D---] FOP.MOV R58, C0
[N--D---] FOP.MOV R59, C0
[N--D---] FOP.MOV R60, C0
[N--D---] FOP.MOV R61, C0
[N--D---] FOP.MOV R62, C0
[N--D---] FOP.MOV R63, C0
[N--D---] FOP.MOV R64, C0
[N--D---] FOP.MOV R65, C0
[N--D---] FOP.MOV R66, C0
[N--D---] FOP.MOV R67, C0
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
[N--D---] FOP.MOV R80, C0
[N--D---] FOP.MOV R81, C0
[N--D---] FOP.MOV R82, C0
[N--D---] FOP.MOV R83, C0
[N--D---] FOP.MOV R84, C0
[N--D---] FOP.MOV R85, C0
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

[NwwD-s-]       LDMA.WR.B128.BST1 R[122:125], R13, X4, 128 , BURST_STRIDE=16 
[N--D-s-]       LDMA.WR.B128.BST1 R[126:129], R13, X4, 384 , BURST_STRIDE=16 
[N--D-s-]       LDMA.WR.B128.BST1 R[130:133], R13, X4, 4224 , BURST_STRIDE=16 
[N--D-s-]       LDMA.WR.B128.BST1 R[134:137], R13, X4, 4480 , BURST_STRIDE=16 
[N--D-s-]       LDMA.WR.B128.BST1 R[138:141], R14, X4, 16512 , BURST_STRIDE=16 
[N-wD---]       INT.ADD R6.s32, SH48.s32, R6.s32
[N-sD-w-]       INT.ADD R7.s32, SH53.s32, R7.s32
[NswD---]       INT.ADD R10.s32, SH95.s32, R10.s32
[NwsD---]       INT.BYP.TL p0, R10.s32, R11.s32
[NwwD---]       CTRL.BR.ANYP0T BB0_no_reset_ic_pre
[NwwD---]       CTRL.BR SL0, reset_ic_idx@musarel32@dword
BB0_no_reset_ic_pre:
[NwwD---]       AP.BARRIER LM0
[N-wD-s-]       LDMA.RD.B32.BST4  R[96:99], R15, X4, 128, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[100:103], R15, X4, 2176, BURST_STRIDE=512
[N--D-o-]       LDMA.RD.B32.BST4  R[113:116], R16, X4, 16512, BURST_STRIDE=512 
[NwsD---]       INT.BYP.TE p0, R8.s32, C0.s32
[N-wD---]   P0    BIT.LUT.AND R113, R113, R142
[N--D---]   P0    BIT.LUT.AND R114, R114, R142
[N--D---]   P0    BIT.LUT.AND R115, R115, R142
[N-sD---]   P0    BIT.LUT.AND R116, R116, R142
[NswD--w]       INT.BYP.TG p0, R12.s32, C0.s32
[Nw-D---]       CTRL.BR.ALLP0F LAST_LOOP


MAIN_LOOP:

[N--D---] MMA.323232 R32[4:2][2:1].s32, R96[2:1][1:1].s8.row, R113[1:1][2:1].s8.col, R32[4:2][2:1].s32
[N--Ds--]       DMA.LD.B512 R[122:137], SH[4:5], R6, _, _, _, _,  SH[4:6]
[N--D---] MMA.323232 R40[4:2][2:1].s32, R96[2:1][1:1].s8.row, R115[1:1][2:1].s8.col, R40[4:2][2:1].s32
[N--Ds--]       DMA.LD.B128 R[138:141], SH[8:9], R7, _, _, _, _,  SH[8:10]
[N--D---] MMA.323232 R48[4:2][2:1].s32, R98[2:1][1:1].s8.row, R113[1:1][2:1].s8.col, R48[4:2][2:1].s32
[N-wD--s]       LDMA.RD.B32.BST4  R[104:107], R15, X4, 4224, BURST_STRIDE=512
[N--D---] MMA.323232 R56[4:2][2:1].s32, R98[2:1][1:1].s8.row, R115[1:1][2:1].s8.col, R56[4:2][2:1].s32
[N--D--s]       LDMA.RD.B32.BST4  R[108:111], R15, X4, 6272, BURST_STRIDE=512
[N--D---] MMA.323232 R64[4:2][2:1].s32, R100[2:1][1:1].s8.row, R113[1:1][2:1].s8.col, R64[4:2][2:1].s32
[N--D--s]       LDMA.RD.B32.BST4  R[117:120], R16, X4, 18560, BURST_STRIDE=512
[N--D---] MMA.323232 R72[4:2][2:1].s32, R100[2:1][1:1].s8.row, R115[1:1][2:1].s8.col, R72[4:2][2:1].s32
[N--D---] MMA.323232 R80[4:2][2:1].s32, R102[2:1][1:1].s8.row, R113[1:1][2:1].s8.col, R80[4:2][2:1].s32
[NwsD--w]       INT.BYP.TE p0, R8.s32, C0.s32
[N-wD---]   P0    BIT.LUT.AND R117, R117, R143
[N--D---]   P0    BIT.LUT.AND R118, R118, R143
[N--D---]   P0    BIT.LUT.AND R119, R119, R143
[Ns-D---]   P0    BIT.LUT.AND R120, R120, R143
[N-sD---] MMA.323232 R88[4:2][2:1].s32, R102[2:1][1:1].s8.row, R115[1:1][2:1].s8.col, R88[4:2][2:1].s32
[Nw-D---]   AP.BARRIER LM0
[Nw-Dw--]   CTRL.NOP
[Nw-D---] MMA.323232 R32[4:2][2:1].s32, R104[2:1][1:1].s8.row, R117[1:1][2:1].s8.col, R32[4:2][2:1].s32
[NwwD-s-] LDMA.WR.B128.BST2 R[122:129], R13, X4, 128, BURST_STRIDE=256
[N--D---] MMA.323232 R40[4:2][2:1].s32, R104[2:1][1:1].s8.row, R119[1:1][2:1].s8.col, R40[4:2][2:1].s32
[N--D-s-] LDMA.WR.B128.BST2 R[130:137], R13, X4, 4224, BURST_STRIDE=256
[N--D---] MMA.323232 R48[4:2][2:1].s32, R106[2:1][1:1].s8.row, R117[1:1][2:1].s8.col, R48[4:2][2:1].s32
[N--D-s-]       LDMA.WR.B128.BST1 R[138:141], R14, X4, 16512 , BURST_STRIDE=16
[N--D---] MMA.323232 R56[4:2][2:1].s32, R106[2:1][1:1].s8.row, R119[1:1][2:1].s8.col, R56[4:2][2:1].s32
[N-wD---]       INT.ADD R6.s32, SH48.s32, R6.s32
[N-sD---]       INT.ADD R7.s32, SH53.s32, R7.s32
[N--D-w-]   CTRL.NOP
[Nw-D---]   AP.BARRIER LM0
[N--D---] MMA.323232 R64[4:2][2:1].s32, R108[2:1][1:1].s8.row, R117[1:1][2:1].s8.col, R64[4:2][2:1].s32
[N-wD--s]       LDMA.RD.B32.BST4  R[96:99], R15, X4, 128, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[100:103], R15, X4, 2176, BURST_STRIDE=512
[N--D---] MMA.323232 R72[4:2][2:1].s32, R108[2:1][1:1].s8.row, R119[1:1][2:1].s8.col, R72[4:2][2:1].s32
[N--D--s]       LDMA.RD.B32.BST4  R[113:116], R16, X4, 16512, BURST_STRIDE=512
[N--D---] MMA.323232 R80[4:2][2:1].s32, R110[2:1][1:1].s8.row, R117[1:1][2:1].s8.col, R80[4:2][2:1].s32
[NwsD--w]       INT.BYP.TL p0, R10.s32, SH57.s32
[NswD---]       INT.BYP.MOVC.PRB R8.s32, C33.s32, C0.s32, p0
[NwsD---]       INT.BYP.TE p0, R8.s32, C0.s32
[N-wD---]   P0    BIT.LUT.AND R113, R113, R142
[N--D---]   P0    BIT.LUT.AND R114, R114, R142
[N--D---]   P0    BIT.LUT.AND R115, R115, R142
[Ns-D---]   P0    BIT.LUT.AND R116, R116, R142
[N-sD---] MMA.323232 R88[4:2][2:1].s32, R110[2:1][1:1].s8.row, R119[1:1][2:1].s8.col, R88[4:2][2:1].s32
[Ns-D---]    INT.ADD R10.s32, SH95.s32, R10.s32
[NwsD---]    INT.BYP.TL p0, R10.s32, R11.s32
[NwwD---]    CTRL.BR.ANYP0T BB0_no_reset_ic
[NwwD---]    CTRL.BR SL0, reset_ic_idx@musarel32@dword
BB0_no_reset_ic:
[Ns-D---]    INT.ADD R12.s32, R12.s32, SH95.neg.s32
[NwsD---]    INT.BYP.TG p0, R12.s32, C0.s32
[N-wD---]    CTRL.BR.ANYP0T MAIN_LOOP


LAST_LOOP:
[N-wD---] MMA.323232 R32[4:2][2:1].s32, R96[2:1][1:1].s8.row, R113[1:1][2:1].s8.col, R32[4:2][2:1].s32
[NswD---]       BIT.ROT.LSR R26, C79, R1
[NwsD---]       BIT.ROT.LSL R26, C45, R26
[NswD---]       BIT.AND.LSR R3, C50, C78, R1
[NwsD---]       INT.ADD R26.s32, R3.s32, R26.s32
[NswD---]       INT.MADD R26.s32, SH104.s32, R26.s32, R0.s32
[NwsD---]       INT.ADD R26.s32, SL2.s32, R26.s32
[N--D---] MMA.323232 R40[4:2][2:1].s32, R96[2:1][1:1].s8.row, R115[1:1][2:1].s8.col, R40[4:2][2:1].s32
[NswD---]       BIT.AND.LSR R12, C80, SH94, R1
[NwsD---]       BIT.ROT.LSL R12, C80, R12
[NswD---]       BIT.AND.LSL R3, C45, C59, R1
[NwsD---]       INT.ADD R12.s32, R3.s32, R12.s32
[NswD---]       INT.ADD R12.s32, SL3.s32, R12.s32
[NwsD---]       INT.ADD R4.s32, SH26.s32, R12.neg.s32
[NswD---]       INT.MADD R12.s32, SH26.s32, R12.s32, R2.s32
[N--D---] MMA.323232 R48[4:2][2:1].s32, R98[2:1][1:1].s8.row, R113[1:1][2:1].s8.col, R48[4:2][2:1].s32
[NwsD---]       INT.MADD R27.s32, SH42.s32, R12.s32, R26.s32
[NswD---]       BIT.ROT.LSL R27, C33, R27
[NwsD---]       FOP.MOV R29, SH42
[NswD---]       INT.ADD R3.s32, SH41.s32, R26.neg.s32
[NwsD---]       INT.MUL R28.s32, SH95.s32,  R29.s32
[N--D---] MMA.323232 R56[4:2][2:1].s32, R98[2:1][1:1].s8.row, R115[1:1][2:1].s8.col, R56[4:2][2:1].s32
[NwwD---]       INT.MADD R152.s32, C0.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R153.s32, C39.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R154.s32, SH90.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R155.s32, SH92.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R156.s32, SH95.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R157.s32, SH97.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R158.s32, SH99.s32, R27.s32, R29.s32
[N--D---] MMA.323232 R64[4:2][2:1].s32, R100[2:1][1:1].s8.row, R113[1:1][2:1].s8.col, R64[4:2][2:1].s32
[N-wD--s]       LDMA.RD.B32.BST4  R[104:107], R15, X4, 4224, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[108:111], R15, X4, 6272, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[117:120], R16, X4, 18560, BURST_STRIDE=512
[N--D---] MMA.323232 R72[4:2][2:1].s32, R100[2:1][1:1].s8.row, R115[1:1][2:1].s8.col, R72[4:2][2:1].s32
[N--D--w]    CTRL.NOP
[N--D---] MMA.323232 R80[4:2][2:1].s32, R102[2:1][1:1].s8.row, R113[1:1][2:1].s8.col, R80[4:2][2:1].s32
[N--D---]       INT.MADD R159.s32, SH101.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R160.s32, SH104.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R161.s32, SH105.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R162.s32, SH106.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R163.s32, SH107.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R164.s32, SH108.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R165.s32, SH109.s32, R27.s32, R29.s32
[N--D---]       INT.MADD R166.s32, SH110.s32, R27.s32, R29.s32
[Ns-D---]       INT.MADD R167.s32, SH111.s32, R27.s32, R29.s32
[Ns-D---] MMA.323232 R88[4:2][2:1].s32, R102[2:1][1:1].s8.row, R115[1:1][2:1].s8.col, R88[4:2][2:1].s32
[NwwD---]       BIT.LUT.AND R117, R117, R143
[N--D---]       BIT.LUT.AND R118, R118, R143
[N--D---]       BIT.LUT.AND R119, R119, R143
[N-sD---]       BIT.LUT.AND R120, R120, R143
[N-wD---] MMA.323232 R32[4:2][2:1].s32, R104[2:1][1:1].s8.row, R117[1:1][2:1].s8.col, R32[4:2][2:1].s32
[N--D---] MMA.323232 R40[4:2][2:1].s32, R104[2:1][1:1].s8.row, R119[1:1][2:1].s8.col, R40[4:2][2:1].s32
[N--D---]       FOP.MOV R144, C0
[N--D---]       FOP.MOV R145, C0
[N--D---]       FOP.MOV R146, C0
[Ns-D---]       FOP.MOV R147, C0
[N--D---] MMA.323232 R48[4:2][2:1].s32, R106[2:1][1:1].s8.row, R117[1:1][2:1].s8.col, R48[4:2][2:1].s32
[N--D---] MMA.323232 R56[4:2][2:1].s32, R106[2:1][1:1].s8.row, R119[1:1][2:1].s8.col, R56[4:2][2:1].s32
[Ns-D---]        INT.BYP.TE p0, SH60.s32, C33.s32
[NwsD---]  P0  BIT.ROT.LSL R144, C45, R12
[N-wD-s-]  P0      DMA.LD.B256 R[144:151], SH[12:13], R144, _, _, _, _,  SH[12:14]
[N--D---] MMA.323232 R64[4:2][2:1].s32, R108[2:1][1:1].s8.row, R117[1:1][2:1].s8.col, R64[4:2][2:1].s32
[N--D---] MMA.323232 R72[4:2][2:1].s32, R108[2:1][1:1].s8.row, R119[1:1][2:1].s8.col, R72[4:2][2:1].s32
[N--D---] MMA.323232 R80[4:2][2:1].s32, R110[2:1][1:1].s8.row, R117[1:1][2:1].s8.col, R80[4:2][2:1].s32
[N-sD-w-] MMA.323232 R88[4:2][2:1].s32, R110[2:1][1:1].s8.row, R119[1:1][2:1].s8.col, R88[4:2][2:1].s32
[N-wD---]       FOP.MUL R144, R144, SH61
[N--D---]       FOP.MUL R145, R145, SH61
[N--D---]       FOP.MUL R146, R146, SH61
[N-sD---]       FOP.MUL R147, R147, SH61
; core loop stage end

; debug use








[NwwD---]     MOV.S32_TO_F32_RTE R32, R32
[N--D---]     MOV.S32_TO_F32_RTE R33, R33
[N--D---]     MOV.S32_TO_F32_RTE R34, R34
[N--D---]     MOV.S32_TO_F32_RTE R35, R35
[N--D---]     MOV.S32_TO_F32_RTE R36, R36
[N--D---]     MOV.S32_TO_F32_RTE R37, R37
[N--D---]     MOV.S32_TO_F32_RTE R38, R38
[N--D---]     MOV.S32_TO_F32_RTE R39, R39
[N--D---]     MOV.S32_TO_F32_RTE R40, R40
[N--D---]     MOV.S32_TO_F32_RTE R41, R41
[N--D---]     MOV.S32_TO_F32_RTE R42, R42
[N--D---]     MOV.S32_TO_F32_RTE R43, R43
[N--D---]     MOV.S32_TO_F32_RTE R44, R44
[N--D---]     MOV.S32_TO_F32_RTE R45, R45
[N--D---]     MOV.S32_TO_F32_RTE R46, R46
[N--D---]     MOV.S32_TO_F32_RTE R47, R47
[N--D---]     MOV.S32_TO_F32_RTE R48, R48
[N--D---]     MOV.S32_TO_F32_RTE R49, R49
[N--D---]     MOV.S32_TO_F32_RTE R50, R50
[N--D---]     MOV.S32_TO_F32_RTE R51, R51
[N--D---]     MOV.S32_TO_F32_RTE R52, R52
[N--D---]     MOV.S32_TO_F32_RTE R53, R53
[N--D---]     MOV.S32_TO_F32_RTE R54, R54
[N--D---]     MOV.S32_TO_F32_RTE R55, R55
[N--D---]     MOV.S32_TO_F32_RTE R56, R56
[N--D---]     MOV.S32_TO_F32_RTE R57, R57
[N--D---]     MOV.S32_TO_F32_RTE R58, R58
[N--D---]     MOV.S32_TO_F32_RTE R59, R59
[N--D---]     MOV.S32_TO_F32_RTE R60, R60
[N--D---]     MOV.S32_TO_F32_RTE R61, R61
[N--D---]     MOV.S32_TO_F32_RTE R62, R62
[N--D---]     MOV.S32_TO_F32_RTE R63, R63
[N--D---]     MOV.S32_TO_F32_RTE R64, R64
[N--D---]     MOV.S32_TO_F32_RTE R65, R65
[N--D---]     MOV.S32_TO_F32_RTE R66, R66
[N--D---]     MOV.S32_TO_F32_RTE R67, R67
[N--D---]     MOV.S32_TO_F32_RTE R68, R68
[N--D---]     MOV.S32_TO_F32_RTE R69, R69
[N--D---]     MOV.S32_TO_F32_RTE R70, R70
[N--D---]     MOV.S32_TO_F32_RTE R71, R71
[N--D---]     MOV.S32_TO_F32_RTE R72, R72
[N--D---]     MOV.S32_TO_F32_RTE R73, R73
[N--D---]     MOV.S32_TO_F32_RTE R74, R74
[N--D---]     MOV.S32_TO_F32_RTE R75, R75
[N--D---]     MOV.S32_TO_F32_RTE R76, R76
[N--D---]     MOV.S32_TO_F32_RTE R77, R77
[N--D---]     MOV.S32_TO_F32_RTE R78, R78
[N--D---]     MOV.S32_TO_F32_RTE R79, R79
[N--D---]     MOV.S32_TO_F32_RTE R80, R80
[N--D---]     MOV.S32_TO_F32_RTE R81, R81
[N--D---]     MOV.S32_TO_F32_RTE R82, R82
[N--D---]     MOV.S32_TO_F32_RTE R83, R83
[N--D---]     MOV.S32_TO_F32_RTE R84, R84
[N--D---]     MOV.S32_TO_F32_RTE R85, R85
[N--D---]     MOV.S32_TO_F32_RTE R86, R86
[N--D---]     MOV.S32_TO_F32_RTE R87, R87
[N--D---]     MOV.S32_TO_F32_RTE R88, R88
[N--D---]     MOV.S32_TO_F32_RTE R89, R89
[N--D---]     MOV.S32_TO_F32_RTE R90, R90
[N--D---]     MOV.S32_TO_F32_RTE R91, R91
[N--D---]     MOV.S32_TO_F32_RTE R92, R92
[N--D---]     MOV.S32_TO_F32_RTE R93, R93
[N--D---]     MOV.S32_TO_F32_RTE R94, R94
[N-sD---]     MOV.S32_TO_F32_RTE R95, R95
[NswD---]       BIT.ROT.LSL R96, C33, R12
[NwsD---]       BIT.ROT.LSL R27, C33, R26
[NwwD-s-]       DMA.LD.B64 R[185:186], SH[72:73], R96, _, _, _, _,  SH[72:74]
[NswD---]       INT.ADD R28.s32, C0.s32, R27.s32
[NwsD---]       INT.ADD R29.s32, C39.s32, R27.s32
[NswD---]       INT.ADD R30.s32, SH90.s32, R27.s32
[NwsD---]       INT.ADD R31.s32, SH92.s32, R27.s32
[NwwD---]       CTRL.NOP
[N--D-s-]       DMA.LD.B16 R169.l, SH[68:69], R28, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R170.l, SH[68:69], R29, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R171.l, SH[68:69], R30, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R172.l, SH[68:69], R31, _, _, _, _,  SH[68:70]
[NwwDwww]       CTRL.NOP
[NswD---]       INT.ADD R28.s32, SH95.s32, R27.s32
[NwsD---]       INT.ADD R29.s32, SH97.s32, R27.s32
[NswD---]       INT.ADD R30.s32, SH99.s32, R27.s32
[NwsD---]       INT.ADD R31.s32, SH101.s32, R27.s32
[NwwD---]       CTRL.NOP
[N--D-s-]       DMA.LD.B16 R173.l, SH[68:69], R28, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R174.l, SH[68:69], R29, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R175.l, SH[68:69], R30, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R176.l, SH[68:69], R31, _, _, _, _,  SH[68:70]
[NwwDwww]       CTRL.NOP
[NswD---]       INT.ADD R28.s32, SH104.s32, R27.s32
[NwsD---]       INT.ADD R29.s32, SH105.s32, R27.s32
[NswD---]       INT.ADD R30.s32, SH106.s32, R27.s32
[NwsD---]       INT.ADD R31.s32, SH107.s32, R27.s32
[NwwD---]       CTRL.NOP
[N--D-s-]       DMA.LD.B16 R177.l, SH[68:69], R28, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R178.l, SH[68:69], R29, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R179.l, SH[68:69], R30, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R180.l, SH[68:69], R31, _, _, _, _,  SH[68:70]
[NwwDwww]       CTRL.NOP
[NswD---]       INT.ADD R28.s32, SH108.s32, R27.s32
[NwsD---]       INT.ADD R29.s32, SH109.s32, R27.s32
[NswD---]       INT.ADD R30.s32, SH110.s32, R27.s32
[NwsD---]       INT.ADD R31.s32, SH111.s32, R27.s32
[NwwD---]       CTRL.NOP
[N--D-s-]       DMA.LD.B16 R181.l, SH[68:69], R28, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R182.l, SH[68:69], R29, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R183.l, SH[68:69], R30, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R184.l, SH[68:69], R31, _, _, _, _,  SH[68:70]
[NwwDwww]       CTRL.NOP
[NswD---]     INT.BYP.TE p0, SH62.s32, C0.s32
[Nw-D---]     CTRL.BR.ANYP0T IDENTITY_START
[NswD---]     INT.BYP.TE p0, SH62.s32, C45.s32
[Nw-D---]     CTRL.BR.ANYP0T RELU_START
[NswD---]     INT.BYP.TE p0, SH62.s32, SH87.s32
[Nw-D---]     CTRL.BR.ANYP0T LEAKYRELU_START
IDENTITY_START:
[NswD---]       INT.BYP.TLE p0, R3.s32, SH102.s32
[NwsD---]       INT.BYP.TLE.OP p0, R4.s32, C59.s32
[NwwD---]        CTRL.BR.ANYP0T STORE_MASK_IDENTITY
[N-wD---] FOP.FMA R32, R32, R185.l, R144
[N--D---] FOP.FMA R33, R33, R185.h, R145
[N--D---] FOP.FMA R40, R40, R186.l, R146
[N-sD---] FOP.FMA R41, R41, R186.h, R147
[N--D---] FOP.MUL R32, R169.l, R32
[N--D---] FOP.MOV R112.l.bf16, R32.f32
[N--D---] FOP.MUL R40, R169.l, R40
[N--D---] FOP.MOV R113.l.bf16, R40.f32
[N--D---] FOP.MUL R33, R169.l, R33
[N--D---] FOP.MOV R112.h.bf16, R33.f32
[N--D---] FOP.MUL R41, R169.l, R41
[N-sD---] FOP.MOV R113.h.bf16, R41.f32
[NwwD---]       DMA.ST.B64 R[112:113], SH[0:1], R152, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R34, R34, R185.l, R144
[N--D---] FOP.FMA R35, R35, R185.h, R145
[N--D---] FOP.FMA R42, R42, R186.l, R146
[N-sD---] FOP.FMA R43, R43, R186.h, R147
[N--D---] FOP.MUL R34, R170.l, R34
[N--D---] FOP.MOV R114.l.bf16, R34.f32
[N--D---] FOP.MUL R42, R170.l, R42
[N--D---] FOP.MOV R115.l.bf16, R42.f32
[N--D---] FOP.MUL R35, R170.l, R35
[N--D---] FOP.MOV R114.h.bf16, R35.f32
[N--D---] FOP.MUL R43, R170.l, R43
[N-sD---] FOP.MOV R115.h.bf16, R43.f32
[NwwD---]       DMA.ST.B64 R[114:115], SH[0:1], R153, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R36, R36, R185.l, R144
[N--D---] FOP.FMA R37, R37, R185.h, R145
[N--D---] FOP.FMA R44, R44, R186.l, R146
[N-sD---] FOP.FMA R45, R45, R186.h, R147
[N--D---] FOP.MUL R36, R171.l, R36
[N--D---] FOP.MOV R116.l.bf16, R36.f32
[N--D---] FOP.MUL R44, R171.l, R44
[N--D---] FOP.MOV R117.l.bf16, R44.f32
[N--D---] FOP.MUL R37, R171.l, R37
[N--D---] FOP.MOV R116.h.bf16, R37.f32
[N--D---] FOP.MUL R45, R171.l, R45
[N-sD---] FOP.MOV R117.h.bf16, R45.f32
[NwwD---]       DMA.ST.B64 R[116:117], SH[0:1], R154, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R38, R38, R185.l, R144
[N--D---] FOP.FMA R39, R39, R185.h, R145
[N--D---] FOP.FMA R46, R46, R186.l, R146
[N-sD---] FOP.FMA R47, R47, R186.h, R147
[N--D---] FOP.MUL R38, R172.l, R38
[N--D---] FOP.MOV R118.l.bf16, R38.f32
[N--D---] FOP.MUL R46, R172.l, R46
[N--D---] FOP.MOV R119.l.bf16, R46.f32
[N--D---] FOP.MUL R39, R172.l, R39
[N--D---] FOP.MOV R118.h.bf16, R39.f32
[N--D---] FOP.MUL R47, R172.l, R47
[N-sD---] FOP.MOV R119.h.bf16, R47.f32
[NwwD---]       DMA.ST.B64 R[118:119], SH[0:1], R155, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R48, R48, R185.l, R144
[N--D---] FOP.FMA R49, R49, R185.h, R145
[N--D---] FOP.FMA R56, R56, R186.l, R146
[N-sD---] FOP.FMA R57, R57, R186.h, R147
[N--D---] FOP.MUL R48, R173.l, R48
[N--D---] FOP.MOV R32.l.bf16, R48.f32
[N--D---] FOP.MUL R56, R173.l, R56
[N--D---] FOP.MOV R33.l.bf16, R56.f32
[N--D---] FOP.MUL R49, R173.l, R49
[N--D---] FOP.MOV R32.h.bf16, R49.f32
[N--D---] FOP.MUL R57, R173.l, R57
[N-sD---] FOP.MOV R33.h.bf16, R57.f32
[NwwD---]       DMA.ST.B64 R[32:33], SH[0:1], R156, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R50, R50, R185.l, R144
[N--D---] FOP.FMA R51, R51, R185.h, R145
[N--D---] FOP.FMA R58, R58, R186.l, R146
[N-sD---] FOP.FMA R59, R59, R186.h, R147
[N--D---] FOP.MUL R50, R174.l, R50
[N--D---] FOP.MOV R36.l.bf16, R50.f32
[N--D---] FOP.MUL R58, R174.l, R58
[N--D---] FOP.MOV R37.l.bf16, R58.f32
[N--D---] FOP.MUL R51, R174.l, R51
[N--D---] FOP.MOV R36.h.bf16, R51.f32
[N--D---] FOP.MUL R59, R174.l, R59
[N-sD---] FOP.MOV R37.h.bf16, R59.f32
[NwwD---]       DMA.ST.B64 R[36:37], SH[0:1], R157, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R52, R52, R185.l, R144
[N--D---] FOP.FMA R53, R53, R185.h, R145
[N--D---] FOP.FMA R60, R60, R186.l, R146
[N-sD---] FOP.FMA R61, R61, R186.h, R147
[N--D---] FOP.MUL R52, R175.l, R52
[N--D---] FOP.MOV R40.l.bf16, R52.f32
[N--D---] FOP.MUL R60, R175.l, R60
[N--D---] FOP.MOV R41.l.bf16, R60.f32
[N--D---] FOP.MUL R53, R175.l, R53
[N--D---] FOP.MOV R40.h.bf16, R53.f32
[N--D---] FOP.MUL R61, R175.l, R61
[N-sD---] FOP.MOV R41.h.bf16, R61.f32
[NwwD---]       DMA.ST.B64 R[40:41], SH[0:1], R158, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R54, R54, R185.l, R144
[N--D---] FOP.FMA R55, R55, R185.h, R145
[N--D---] FOP.FMA R62, R62, R186.l, R146
[N-sD---] FOP.FMA R63, R63, R186.h, R147
[N--D---] FOP.MUL R54, R176.l, R54
[N--D---] FOP.MOV R44.l.bf16, R54.f32
[N--D---] FOP.MUL R62, R176.l, R62
[N--D---] FOP.MOV R45.l.bf16, R62.f32
[N--D---] FOP.MUL R55, R176.l, R55
[N--D---] FOP.MOV R44.h.bf16, R55.f32
[N--D---] FOP.MUL R63, R176.l, R63
[N-sD---] FOP.MOV R45.h.bf16, R63.f32
[NwwD---]       DMA.ST.B64 R[44:45], SH[0:1], R159, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R64, R64, R185.l, R144
[N--D---] FOP.FMA R65, R65, R185.h, R145
[N--D---] FOP.FMA R72, R72, R186.l, R146
[N-sD---] FOP.FMA R73, R73, R186.h, R147
[N--D---] FOP.MUL R64, R177.l, R64
[N--D---] FOP.MOV R48.l.bf16, R64.f32
[N--D---] FOP.MUL R72, R177.l, R72
[N--D---] FOP.MOV R49.l.bf16, R72.f32
[N--D---] FOP.MUL R65, R177.l, R65
[N--D---] FOP.MOV R48.h.bf16, R65.f32
[N--D---] FOP.MUL R73, R177.l, R73
[N-sD---] FOP.MOV R49.h.bf16, R73.f32
[NwwD---]       DMA.ST.B64 R[48:49], SH[0:1], R160, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R66, R66, R185.l, R144
[N--D---] FOP.FMA R67, R67, R185.h, R145
[N--D---] FOP.FMA R74, R74, R186.l, R146
[N-sD---] FOP.FMA R75, R75, R186.h, R147
[N--D---] FOP.MUL R66, R178.l, R66
[N--D---] FOP.MOV R52.l.bf16, R66.f32
[N--D---] FOP.MUL R74, R178.l, R74
[N--D---] FOP.MOV R53.l.bf16, R74.f32
[N--D---] FOP.MUL R67, R178.l, R67
[N--D---] FOP.MOV R52.h.bf16, R67.f32
[N--D---] FOP.MUL R75, R178.l, R75
[N-sD---] FOP.MOV R53.h.bf16, R75.f32
[NwwD---]       DMA.ST.B64 R[52:53], SH[0:1], R161, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R68, R68, R185.l, R144
[N--D---] FOP.FMA R69, R69, R185.h, R145
[N--D---] FOP.FMA R76, R76, R186.l, R146
[N-sD---] FOP.FMA R77, R77, R186.h, R147
[N--D---] FOP.MUL R68, R179.l, R68
[N--D---] FOP.MOV R56.l.bf16, R68.f32
[N--D---] FOP.MUL R76, R179.l, R76
[N--D---] FOP.MOV R57.l.bf16, R76.f32
[N--D---] FOP.MUL R69, R179.l, R69
[N--D---] FOP.MOV R56.h.bf16, R69.f32
[N--D---] FOP.MUL R77, R179.l, R77
[N-sD---] FOP.MOV R57.h.bf16, R77.f32
[NwwD---]       DMA.ST.B64 R[56:57], SH[0:1], R162, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R70, R70, R185.l, R144
[N--D---] FOP.FMA R71, R71, R185.h, R145
[N--D---] FOP.FMA R78, R78, R186.l, R146
[N-sD---] FOP.FMA R79, R79, R186.h, R147
[N--D---] FOP.MUL R70, R180.l, R70
[N--D---] FOP.MOV R60.l.bf16, R70.f32
[N--D---] FOP.MUL R78, R180.l, R78
[N--D---] FOP.MOV R61.l.bf16, R78.f32
[N--D---] FOP.MUL R71, R180.l, R71
[N--D---] FOP.MOV R60.h.bf16, R71.f32
[N--D---] FOP.MUL R79, R180.l, R79
[N-sD---] FOP.MOV R61.h.bf16, R79.f32
[NwwD---]       DMA.ST.B64 R[60:61], SH[0:1], R163, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R80, R80, R185.l, R144
[N--D---] FOP.FMA R81, R81, R185.h, R145
[N--D---] FOP.FMA R88, R88, R186.l, R146
[N-sD---] FOP.FMA R89, R89, R186.h, R147
[N--D---] FOP.MUL R80, R181.l, R80
[N--D---] FOP.MOV R64.l.bf16, R80.f32
[N--D---] FOP.MUL R88, R181.l, R88
[N--D---] FOP.MOV R65.l.bf16, R88.f32
[N--D---] FOP.MUL R81, R181.l, R81
[N--D---] FOP.MOV R64.h.bf16, R81.f32
[N--D---] FOP.MUL R89, R181.l, R89
[N-sD---] FOP.MOV R65.h.bf16, R89.f32
[NwwD---]       DMA.ST.B64 R[64:65], SH[0:1], R164, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R82, R82, R185.l, R144
[N--D---] FOP.FMA R83, R83, R185.h, R145
[N--D---] FOP.FMA R90, R90, R186.l, R146
[N-sD---] FOP.FMA R91, R91, R186.h, R147
[N--D---] FOP.MUL R82, R182.l, R82
[N--D---] FOP.MOV R68.l.bf16, R82.f32
[N--D---] FOP.MUL R90, R182.l, R90
[N--D---] FOP.MOV R69.l.bf16, R90.f32
[N--D---] FOP.MUL R83, R182.l, R83
[N--D---] FOP.MOV R68.h.bf16, R83.f32
[N--D---] FOP.MUL R91, R182.l, R91
[N-sD---] FOP.MOV R69.h.bf16, R91.f32
[NwwD---]       DMA.ST.B64 R[68:69], SH[0:1], R165, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R84, R84, R185.l, R144
[N--D---] FOP.FMA R85, R85, R185.h, R145
[N--D---] FOP.FMA R92, R92, R186.l, R146
[N-sD---] FOP.FMA R93, R93, R186.h, R147
[N--D---] FOP.MUL R84, R183.l, R84
[N--D---] FOP.MOV R72.l.bf16, R84.f32
[N--D---] FOP.MUL R92, R183.l, R92
[N--D---] FOP.MOV R73.l.bf16, R92.f32
[N--D---] FOP.MUL R85, R183.l, R85
[N--D---] FOP.MOV R72.h.bf16, R85.f32
[N--D---] FOP.MUL R93, R183.l, R93
[N-sD---] FOP.MOV R73.h.bf16, R93.f32
[NwwD---]       DMA.ST.B64 R[72:73], SH[0:1], R166, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R86, R86, R185.l, R144
[N--D---] FOP.FMA R87, R87, R185.h, R145
[N--D---] FOP.FMA R94, R94, R186.l, R146
[N-sD---] FOP.FMA R95, R95, R186.h, R147
[N--D---] FOP.MUL R86, R184.l, R86
[N--D---] FOP.MOV R76.l.bf16, R86.f32
[N--D---] FOP.MUL R94, R184.l, R94
[N--D---] FOP.MOV R77.l.bf16, R94.f32
[N--D---] FOP.MUL R87, R184.l, R87
[N--D---] FOP.MOV R76.h.bf16, R87.f32
[N--D---] FOP.MUL R95, R184.l, R95
[N-sD---] FOP.MOV R77.h.bf16, R95.f32
[NswD---]       DMA.ST.B64 R[76:77], SH[0:1], R167, _, _, _, _,  SH[0:2]
[NwwDwww]    CTRL.NOP
[NwwD---]    AP.BARRIER LM0
[NwwD---]    CTRL.END
STORE_MASK_IDENTITY:
[NwwD---] FOP.FMA R32, R32, R185.l, R144
[N--D---] FOP.FMA R33, R33, R185.h, R145
[N--D---] FOP.FMA R40, R40, R186.l, R146
[N-sD---] FOP.FMA R41, R41, R186.h, R147
[NwwD---] FOP.MUL R32, R32, R169.l
[N--D---] FOP.MOV R32.l.bf16, R32.f32
[N--D---] FOP.MUL R33, R33, R169.l
[N--D---] FOP.MOV R33.l.bf16, R33.f32
[N--D---] FOP.MUL R40, R40, R169.l
[N--D---] FOP.MOV R40.l.bf16, R40.f32
[N--D---] FOP.MUL R41, R41, R169.l
[N-sD---] FOP.MOV R41.l.bf16, R41.f32
[NswD---]       INT.BYP.TG p0, R3.s32, C0.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R32.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R33.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R40.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R41.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NwwD---] FOP.FMA R34, R34, R185.l, R144
[N--D---] FOP.FMA R35, R35, R185.h, R145
[N--D---] FOP.FMA R42, R42, R186.l, R146
[N-sD---] FOP.FMA R43, R43, R186.h, R147
[NwwD---] FOP.MUL R34, R34, R170.l
[N--D---] FOP.MOV R34.l.bf16, R34.f32
[N--D---] FOP.MUL R35, R35, R170.l
[N--D---] FOP.MOV R35.l.bf16, R35.f32
[N--D---] FOP.MUL R42, R42, R170.l
[N--D---] FOP.MOV R42.l.bf16, R42.f32
[N--D---] FOP.MUL R43, R43, R170.l
[N-sD---] FOP.MOV R43.l.bf16, R43.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, C46.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R34.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R35.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R42.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R43.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NwwD---] FOP.FMA R36, R36, R185.l, R144
[N--D---] FOP.FMA R37, R37, R185.h, R145
[N--D---] FOP.FMA R44, R44, R186.l, R146
[N-sD---] FOP.FMA R45, R45, R186.h, R147
[NwwD---] FOP.MUL R36, R36, R171.l
[N--D---] FOP.MOV R36.l.bf16, R36.f32
[N--D---] FOP.MUL R37, R37, R171.l
[N--D---] FOP.MOV R37.l.bf16, R37.f32
[N--D---] FOP.MUL R44, R44, R171.l
[N--D---] FOP.MOV R44.l.bf16, R44.f32
[N--D---] FOP.MUL R45, R45, R171.l
[N-sD---] FOP.MOV R45.l.bf16, R45.f32
[NswD---]       INT.BYP.TG p0, R3.s32, C39.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R36.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R37.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R44.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R45.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NwwD---] FOP.FMA R38, R38, R185.l, R144
[N--D---] FOP.FMA R39, R39, R185.h, R145
[N--D---] FOP.FMA R46, R46, R186.l, R146
[N-sD---] FOP.FMA R47, R47, R186.h, R147
[NwwD---] FOP.MUL R38, R38, R172.l
[N--D---] FOP.MOV R38.l.bf16, R38.f32
[N--D---] FOP.MUL R39, R39, R172.l
[N--D---] FOP.MOV R39.l.bf16, R39.f32
[N--D---] FOP.MUL R46, R46, R172.l
[N--D---] FOP.MOV R46.l.bf16, R46.f32
[N--D---] FOP.MUL R47, R47, R172.l
[N-sD---] FOP.MOV R47.l.bf16, R47.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, C48.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R38.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R39.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R46.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R47.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NwwD---] FOP.FMA R48, R48, R185.l, R144
[N--D---] FOP.FMA R49, R49, R185.h, R145
[N--D---] FOP.FMA R56, R56, R186.l, R146
[N-sD---] FOP.FMA R57, R57, R186.h, R147
[NwwD---] FOP.MUL R48, R48, R173.l
[N--D---] FOP.MOV R48.l.bf16, R48.f32
[N--D---] FOP.MUL R49, R49, R173.l
[N--D---] FOP.MOV R49.l.bf16, R49.f32
[N--D---] FOP.MUL R56, R56, R173.l
[N--D---] FOP.MOV R56.l.bf16, R56.f32
[N--D---] FOP.MUL R57, R57, R173.l
[N-sD---] FOP.MOV R57.l.bf16, R57.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH90.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R48.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R49.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R56.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R57.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NwwD---] FOP.FMA R50, R50, R185.l, R144
[N--D---] FOP.FMA R51, R51, R185.h, R145
[N--D---] FOP.FMA R58, R58, R186.l, R146
[N-sD---] FOP.FMA R59, R59, R186.h, R147
[NwwD---] FOP.MUL R50, R50, R174.l
[N--D---] FOP.MOV R50.l.bf16, R50.f32
[N--D---] FOP.MUL R51, R51, R174.l
[N--D---] FOP.MOV R51.l.bf16, R51.f32
[N--D---] FOP.MUL R58, R58, R174.l
[N--D---] FOP.MOV R58.l.bf16, R58.f32
[N--D---] FOP.MUL R59, R59, R174.l
[N-sD---] FOP.MOV R59.l.bf16, R59.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH91.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R50.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R51.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R58.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R59.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NwwD---] FOP.FMA R52, R52, R185.l, R144
[N--D---] FOP.FMA R53, R53, R185.h, R145
[N--D---] FOP.FMA R60, R60, R186.l, R146
[N-sD---] FOP.FMA R61, R61, R186.h, R147
[NwwD---] FOP.MUL R52, R52, R175.l
[N--D---] FOP.MOV R52.l.bf16, R52.f32
[N--D---] FOP.MUL R53, R53, R175.l
[N--D---] FOP.MOV R53.l.bf16, R53.f32
[N--D---] FOP.MUL R60, R60, R175.l
[N--D---] FOP.MOV R60.l.bf16, R60.f32
[N--D---] FOP.MUL R61, R61, R175.l
[N-sD---] FOP.MOV R61.l.bf16, R61.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH92.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R52.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R53.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R60.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R61.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NwwD---] FOP.FMA R54, R54, R185.l, R144
[N--D---] FOP.FMA R55, R55, R185.h, R145
[N--D---] FOP.FMA R62, R62, R186.l, R146
[N-sD---] FOP.FMA R63, R63, R186.h, R147
[NwwD---] FOP.MUL R54, R54, R176.l
[N--D---] FOP.MOV R54.l.bf16, R54.f32
[N--D---] FOP.MUL R55, R55, R176.l
[N--D---] FOP.MOV R55.l.bf16, R55.f32
[N--D---] FOP.MUL R62, R62, R176.l
[N--D---] FOP.MOV R62.l.bf16, R62.f32
[N--D---] FOP.MUL R63, R63, R176.l
[N-sD---] FOP.MOV R63.l.bf16, R63.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH93.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R54.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R55.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R62.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R63.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NwwD---] FOP.FMA R64, R64, R185.l, R144
[N--D---] FOP.FMA R65, R65, R185.h, R145
[N--D---] FOP.FMA R72, R72, R186.l, R146
[N-sD---] FOP.FMA R73, R73, R186.h, R147
[NwwD---] FOP.MUL R64, R64, R177.l
[N--D---] FOP.MOV R64.l.bf16, R64.f32
[N--D---] FOP.MUL R65, R65, R177.l
[N--D---] FOP.MOV R65.l.bf16, R65.f32
[N--D---] FOP.MUL R72, R72, R177.l
[N--D---] FOP.MOV R72.l.bf16, R72.f32
[N--D---] FOP.MUL R73, R73, R177.l
[N-sD---] FOP.MOV R73.l.bf16, R73.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH95.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R64.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R65.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R72.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R73.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NwwD---] FOP.FMA R66, R66, R185.l, R144
[N--D---] FOP.FMA R67, R67, R185.h, R145
[N--D---] FOP.FMA R74, R74, R186.l, R146
[N-sD---] FOP.FMA R75, R75, R186.h, R147
[NwwD---] FOP.MUL R66, R66, R178.l
[N--D---] FOP.MOV R66.l.bf16, R66.f32
[N--D---] FOP.MUL R67, R67, R178.l
[N--D---] FOP.MOV R67.l.bf16, R67.f32
[N--D---] FOP.MUL R74, R74, R178.l
[N--D---] FOP.MOV R74.l.bf16, R74.f32
[N--D---] FOP.MUL R75, R75, R178.l
[N-sD---] FOP.MOV R75.l.bf16, R75.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH96.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R66.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R67.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R74.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R75.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NwwD---] FOP.FMA R68, R68, R185.l, R144
[N--D---] FOP.FMA R69, R69, R185.h, R145
[N--D---] FOP.FMA R76, R76, R186.l, R146
[N-sD---] FOP.FMA R77, R77, R186.h, R147
[NwwD---] FOP.MUL R68, R68, R179.l
[N--D---] FOP.MOV R68.l.bf16, R68.f32
[N--D---] FOP.MUL R69, R69, R179.l
[N--D---] FOP.MOV R69.l.bf16, R69.f32
[N--D---] FOP.MUL R76, R76, R179.l
[N--D---] FOP.MOV R76.l.bf16, R76.f32
[N--D---] FOP.MUL R77, R77, R179.l
[N-sD---] FOP.MOV R77.l.bf16, R77.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH97.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R68.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R69.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R76.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R77.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NwwD---] FOP.FMA R70, R70, R185.l, R144
[N--D---] FOP.FMA R71, R71, R185.h, R145
[N--D---] FOP.FMA R78, R78, R186.l, R146
[N-sD---] FOP.FMA R79, R79, R186.h, R147
[NwwD---] FOP.MUL R70, R70, R180.l
[N--D---] FOP.MOV R70.l.bf16, R70.f32
[N--D---] FOP.MUL R71, R71, R180.l
[N--D---] FOP.MOV R71.l.bf16, R71.f32
[N--D---] FOP.MUL R78, R78, R180.l
[N--D---] FOP.MOV R78.l.bf16, R78.f32
[N--D---] FOP.MUL R79, R79, R180.l
[N-sD---] FOP.MOV R79.l.bf16, R79.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH98.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R70.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R71.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R78.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R79.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NwwD---] FOP.FMA R80, R80, R185.l, R144
[N--D---] FOP.FMA R81, R81, R185.h, R145
[N--D---] FOP.FMA R88, R88, R186.l, R146
[N-sD---] FOP.FMA R89, R89, R186.h, R147
[NwwD---] FOP.MUL R80, R80, R181.l
[N--D---] FOP.MOV R80.l.bf16, R80.f32
[N--D---] FOP.MUL R81, R81, R181.l
[N--D---] FOP.MOV R81.l.bf16, R81.f32
[N--D---] FOP.MUL R88, R88, R181.l
[N--D---] FOP.MOV R88.l.bf16, R88.f32
[N--D---] FOP.MUL R89, R89, R181.l
[N-sD---] FOP.MOV R89.l.bf16, R89.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH99.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R80.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R81.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R88.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R89.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NwwD---] FOP.FMA R82, R82, R185.l, R144
[N--D---] FOP.FMA R83, R83, R185.h, R145
[N--D---] FOP.FMA R90, R90, R186.l, R146
[N-sD---] FOP.FMA R91, R91, R186.h, R147
[NwwD---] FOP.MUL R82, R82, R182.l
[N--D---] FOP.MOV R82.l.bf16, R82.f32
[N--D---] FOP.MUL R83, R83, R182.l
[N--D---] FOP.MOV R83.l.bf16, R83.f32
[N--D---] FOP.MUL R90, R90, R182.l
[N--D---] FOP.MOV R90.l.bf16, R90.f32
[N--D---] FOP.MUL R91, R91, R182.l
[N-sD---] FOP.MOV R91.l.bf16, R91.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH100.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R82.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R83.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R90.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R91.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NwwD---] FOP.FMA R84, R84, R185.l, R144
[N--D---] FOP.FMA R85, R85, R185.h, R145
[N--D---] FOP.FMA R92, R92, R186.l, R146
[N-sD---] FOP.FMA R93, R93, R186.h, R147
[NwwD---] FOP.MUL R84, R84, R183.l
[N--D---] FOP.MOV R84.l.bf16, R84.f32
[N--D---] FOP.MUL R85, R85, R183.l
[N--D---] FOP.MOV R85.l.bf16, R85.f32
[N--D---] FOP.MUL R92, R92, R183.l
[N--D---] FOP.MOV R92.l.bf16, R92.f32
[N--D---] FOP.MUL R93, R93, R183.l
[N-sD---] FOP.MOV R93.l.bf16, R93.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH101.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R84.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R85.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R92.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R93.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NwwD---] FOP.FMA R86, R86, R185.l, R144
[N--D---] FOP.FMA R87, R87, R185.h, R145
[N--D---] FOP.FMA R94, R94, R186.l, R146
[N-sD---] FOP.FMA R95, R95, R186.h, R147
[NwwD---] FOP.MUL R86, R86, R184.l
[N--D---] FOP.MOV R86.l.bf16, R86.f32
[N--D---] FOP.MUL R87, R87, R184.l
[N--D---] FOP.MOV R87.l.bf16, R87.f32
[N--D---] FOP.MUL R94, R94, R184.l
[N--D---] FOP.MOV R94.l.bf16, R94.f32
[N--D---] FOP.MUL R95, R95, R184.l
[N-sD---] FOP.MOV R95.l.bf16, R95.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH102.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R86.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R87.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R94.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R95.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NwwDwww]    CTRL.NOP
[NwwD---]    AP.BARRIER LM0
[NwwD---]    CTRL.END
RELU_START:
[NswD---]       INT.BYP.TLE p0, R3.s32, SH102.s32
[NwsD---]       INT.BYP.TLE.OP p0, R4.s32, C59.s32
[NwwD---]        CTRL.BR.ANYP0T STORE_MASK_RELU
[N-wD---] FOP.FMA R32, R32, R185.l, R144
[N--D---] FOP.FMA R33, R33, R185.h, R145
[N--D---] FOP.FMA R40, R40, R186.l, R146
[N-sD---] FOP.FMA R41, R41, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R32, R32, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R33, R33, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R40, R40, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R41, R41, C0
[N--D---] FOP.MUL R32, R169.l, R32
[N--D---] FOP.MOV R112.l.bf16, R32.f32
[N--D---] FOP.MUL R40, R169.l, R40
[N--D---] FOP.MOV R113.l.bf16, R40.f32
[N--D---] FOP.MUL R33, R169.l, R33
[N--D---] FOP.MOV R112.h.bf16, R33.f32
[N--D---] FOP.MUL R41, R169.l, R41
[N-sD---] FOP.MOV R113.h.bf16, R41.f32
[NwwD---]       DMA.ST.B64 R[112:113], SH[0:1], R152, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R34, R34, R185.l, R144
[N--D---] FOP.FMA R35, R35, R185.h, R145
[N--D---] FOP.FMA R42, R42, R186.l, R146
[N-sD---] FOP.FMA R43, R43, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R34, R34, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R35, R35, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R42, R42, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R43, R43, C0
[N--D---] FOP.MUL R34, R170.l, R34
[N--D---] FOP.MOV R114.l.bf16, R34.f32
[N--D---] FOP.MUL R42, R170.l, R42
[N--D---] FOP.MOV R115.l.bf16, R42.f32
[N--D---] FOP.MUL R35, R170.l, R35
[N--D---] FOP.MOV R114.h.bf16, R35.f32
[N--D---] FOP.MUL R43, R170.l, R43
[N-sD---] FOP.MOV R115.h.bf16, R43.f32
[NwwD---]       DMA.ST.B64 R[114:115], SH[0:1], R153, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R36, R36, R185.l, R144
[N--D---] FOP.FMA R37, R37, R185.h, R145
[N--D---] FOP.FMA R44, R44, R186.l, R146
[N-sD---] FOP.FMA R45, R45, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R36, R36, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R37, R37, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R44, R44, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R45, R45, C0
[N--D---] FOP.MUL R36, R171.l, R36
[N--D---] FOP.MOV R116.l.bf16, R36.f32
[N--D---] FOP.MUL R44, R171.l, R44
[N--D---] FOP.MOV R117.l.bf16, R44.f32
[N--D---] FOP.MUL R37, R171.l, R37
[N--D---] FOP.MOV R116.h.bf16, R37.f32
[N--D---] FOP.MUL R45, R171.l, R45
[N-sD---] FOP.MOV R117.h.bf16, R45.f32
[NwwD---]       DMA.ST.B64 R[116:117], SH[0:1], R154, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R38, R38, R185.l, R144
[N--D---] FOP.FMA R39, R39, R185.h, R145
[N--D---] FOP.FMA R46, R46, R186.l, R146
[N-sD---] FOP.FMA R47, R47, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R38, R38, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R39, R39, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R46, R46, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R47, R47, C0
[N--D---] FOP.MUL R38, R172.l, R38
[N--D---] FOP.MOV R118.l.bf16, R38.f32
[N--D---] FOP.MUL R46, R172.l, R46
[N--D---] FOP.MOV R119.l.bf16, R46.f32
[N--D---] FOP.MUL R39, R172.l, R39
[N--D---] FOP.MOV R118.h.bf16, R39.f32
[N--D---] FOP.MUL R47, R172.l, R47
[N-sD---] FOP.MOV R119.h.bf16, R47.f32
[NwwD---]       DMA.ST.B64 R[118:119], SH[0:1], R155, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R48, R48, R185.l, R144
[N--D---] FOP.FMA R49, R49, R185.h, R145
[N--D---] FOP.FMA R56, R56, R186.l, R146
[N-sD---] FOP.FMA R57, R57, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R48, R48, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R49, R49, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R56, R56, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R57, R57, C0
[N--D---] FOP.MUL R48, R173.l, R48
[N--D---] FOP.MOV R32.l.bf16, R48.f32
[N--D---] FOP.MUL R56, R173.l, R56
[N--D---] FOP.MOV R33.l.bf16, R56.f32
[N--D---] FOP.MUL R49, R173.l, R49
[N--D---] FOP.MOV R32.h.bf16, R49.f32
[N--D---] FOP.MUL R57, R173.l, R57
[N-sD---] FOP.MOV R33.h.bf16, R57.f32
[NwwD---]       DMA.ST.B64 R[32:33], SH[0:1], R156, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R50, R50, R185.l, R144
[N--D---] FOP.FMA R51, R51, R185.h, R145
[N--D---] FOP.FMA R58, R58, R186.l, R146
[N-sD---] FOP.FMA R59, R59, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R50, R50, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R51, R51, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R58, R58, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R59, R59, C0
[N--D---] FOP.MUL R50, R174.l, R50
[N--D---] FOP.MOV R36.l.bf16, R50.f32
[N--D---] FOP.MUL R58, R174.l, R58
[N--D---] FOP.MOV R37.l.bf16, R58.f32
[N--D---] FOP.MUL R51, R174.l, R51
[N--D---] FOP.MOV R36.h.bf16, R51.f32
[N--D---] FOP.MUL R59, R174.l, R59
[N-sD---] FOP.MOV R37.h.bf16, R59.f32
[NwwD---]       DMA.ST.B64 R[36:37], SH[0:1], R157, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R52, R52, R185.l, R144
[N--D---] FOP.FMA R53, R53, R185.h, R145
[N--D---] FOP.FMA R60, R60, R186.l, R146
[N-sD---] FOP.FMA R61, R61, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R52, R52, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R53, R53, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R60, R60, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R61, R61, C0
[N--D---] FOP.MUL R52, R175.l, R52
[N--D---] FOP.MOV R40.l.bf16, R52.f32
[N--D---] FOP.MUL R60, R175.l, R60
[N--D---] FOP.MOV R41.l.bf16, R60.f32
[N--D---] FOP.MUL R53, R175.l, R53
[N--D---] FOP.MOV R40.h.bf16, R53.f32
[N--D---] FOP.MUL R61, R175.l, R61
[N-sD---] FOP.MOV R41.h.bf16, R61.f32
[NwwD---]       DMA.ST.B64 R[40:41], SH[0:1], R158, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R54, R54, R185.l, R144
[N--D---] FOP.FMA R55, R55, R185.h, R145
[N--D---] FOP.FMA R62, R62, R186.l, R146
[N-sD---] FOP.FMA R63, R63, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R54, R54, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R55, R55, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R62, R62, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R63, R63, C0
[N--D---] FOP.MUL R54, R176.l, R54
[N--D---] FOP.MOV R44.l.bf16, R54.f32
[N--D---] FOP.MUL R62, R176.l, R62
[N--D---] FOP.MOV R45.l.bf16, R62.f32
[N--D---] FOP.MUL R55, R176.l, R55
[N--D---] FOP.MOV R44.h.bf16, R55.f32
[N--D---] FOP.MUL R63, R176.l, R63
[N-sD---] FOP.MOV R45.h.bf16, R63.f32
[NwwD---]       DMA.ST.B64 R[44:45], SH[0:1], R159, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R64, R64, R185.l, R144
[N--D---] FOP.FMA R65, R65, R185.h, R145
[N--D---] FOP.FMA R72, R72, R186.l, R146
[N-sD---] FOP.FMA R73, R73, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R64, R64, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R65, R65, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R72, R72, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R73, R73, C0
[N--D---] FOP.MUL R64, R177.l, R64
[N--D---] FOP.MOV R48.l.bf16, R64.f32
[N--D---] FOP.MUL R72, R177.l, R72
[N--D---] FOP.MOV R49.l.bf16, R72.f32
[N--D---] FOP.MUL R65, R177.l, R65
[N--D---] FOP.MOV R48.h.bf16, R65.f32
[N--D---] FOP.MUL R73, R177.l, R73
[N-sD---] FOP.MOV R49.h.bf16, R73.f32
[NwwD---]       DMA.ST.B64 R[48:49], SH[0:1], R160, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R66, R66, R185.l, R144
[N--D---] FOP.FMA R67, R67, R185.h, R145
[N--D---] FOP.FMA R74, R74, R186.l, R146
[N-sD---] FOP.FMA R75, R75, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R66, R66, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R67, R67, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R74, R74, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R75, R75, C0
[N--D---] FOP.MUL R66, R178.l, R66
[N--D---] FOP.MOV R52.l.bf16, R66.f32
[N--D---] FOP.MUL R74, R178.l, R74
[N--D---] FOP.MOV R53.l.bf16, R74.f32
[N--D---] FOP.MUL R67, R178.l, R67
[N--D---] FOP.MOV R52.h.bf16, R67.f32
[N--D---] FOP.MUL R75, R178.l, R75
[N-sD---] FOP.MOV R53.h.bf16, R75.f32
[NwwD---]       DMA.ST.B64 R[52:53], SH[0:1], R161, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R68, R68, R185.l, R144
[N--D---] FOP.FMA R69, R69, R185.h, R145
[N--D---] FOP.FMA R76, R76, R186.l, R146
[N-sD---] FOP.FMA R77, R77, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R68, R68, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R69, R69, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R76, R76, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R77, R77, C0
[N--D---] FOP.MUL R68, R179.l, R68
[N--D---] FOP.MOV R56.l.bf16, R68.f32
[N--D---] FOP.MUL R76, R179.l, R76
[N--D---] FOP.MOV R57.l.bf16, R76.f32
[N--D---] FOP.MUL R69, R179.l, R69
[N--D---] FOP.MOV R56.h.bf16, R69.f32
[N--D---] FOP.MUL R77, R179.l, R77
[N-sD---] FOP.MOV R57.h.bf16, R77.f32
[NwwD---]       DMA.ST.B64 R[56:57], SH[0:1], R162, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R70, R70, R185.l, R144
[N--D---] FOP.FMA R71, R71, R185.h, R145
[N--D---] FOP.FMA R78, R78, R186.l, R146
[N-sD---] FOP.FMA R79, R79, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R70, R70, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R71, R71, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R78, R78, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R79, R79, C0
[N--D---] FOP.MUL R70, R180.l, R70
[N--D---] FOP.MOV R60.l.bf16, R70.f32
[N--D---] FOP.MUL R78, R180.l, R78
[N--D---] FOP.MOV R61.l.bf16, R78.f32
[N--D---] FOP.MUL R71, R180.l, R71
[N--D---] FOP.MOV R60.h.bf16, R71.f32
[N--D---] FOP.MUL R79, R180.l, R79
[N-sD---] FOP.MOV R61.h.bf16, R79.f32
[NwwD---]       DMA.ST.B64 R[60:61], SH[0:1], R163, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R80, R80, R185.l, R144
[N--D---] FOP.FMA R81, R81, R185.h, R145
[N--D---] FOP.FMA R88, R88, R186.l, R146
[N-sD---] FOP.FMA R89, R89, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R80, R80, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R81, R81, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R88, R88, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R89, R89, C0
[N--D---] FOP.MUL R80, R181.l, R80
[N--D---] FOP.MOV R64.l.bf16, R80.f32
[N--D---] FOP.MUL R88, R181.l, R88
[N--D---] FOP.MOV R65.l.bf16, R88.f32
[N--D---] FOP.MUL R81, R181.l, R81
[N--D---] FOP.MOV R64.h.bf16, R81.f32
[N--D---] FOP.MUL R89, R181.l, R89
[N-sD---] FOP.MOV R65.h.bf16, R89.f32
[NwwD---]       DMA.ST.B64 R[64:65], SH[0:1], R164, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R82, R82, R185.l, R144
[N--D---] FOP.FMA R83, R83, R185.h, R145
[N--D---] FOP.FMA R90, R90, R186.l, R146
[N-sD---] FOP.FMA R91, R91, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R82, R82, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R83, R83, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R90, R90, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R91, R91, C0
[N--D---] FOP.MUL R82, R182.l, R82
[N--D---] FOP.MOV R68.l.bf16, R82.f32
[N--D---] FOP.MUL R90, R182.l, R90
[N--D---] FOP.MOV R69.l.bf16, R90.f32
[N--D---] FOP.MUL R83, R182.l, R83
[N--D---] FOP.MOV R68.h.bf16, R83.f32
[N--D---] FOP.MUL R91, R182.l, R91
[N-sD---] FOP.MOV R69.h.bf16, R91.f32
[NwwD---]       DMA.ST.B64 R[68:69], SH[0:1], R165, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R84, R84, R185.l, R144
[N--D---] FOP.FMA R85, R85, R185.h, R145
[N--D---] FOP.FMA R92, R92, R186.l, R146
[N-sD---] FOP.FMA R93, R93, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R84, R84, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R85, R85, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R92, R92, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R93, R93, C0
[N--D---] FOP.MUL R84, R183.l, R84
[N--D---] FOP.MOV R72.l.bf16, R84.f32
[N--D---] FOP.MUL R92, R183.l, R92
[N--D---] FOP.MOV R73.l.bf16, R92.f32
[N--D---] FOP.MUL R85, R183.l, R85
[N--D---] FOP.MOV R72.h.bf16, R85.f32
[N--D---] FOP.MUL R93, R183.l, R93
[N-sD---] FOP.MOV R73.h.bf16, R93.f32
[NwwD---]       DMA.ST.B64 R[72:73], SH[0:1], R166, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R86, R86, R185.l, R144
[N--D---] FOP.FMA R87, R87, R185.h, R145
[N--D---] FOP.FMA R94, R94, R186.l, R146
[N-sD---] FOP.FMA R95, R95, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R86, R86, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R87, R87, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R94, R94, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R95, R95, C0
[N--D---] FOP.MUL R86, R184.l, R86
[N--D---] FOP.MOV R76.l.bf16, R86.f32
[N--D---] FOP.MUL R94, R184.l, R94
[N--D---] FOP.MOV R77.l.bf16, R94.f32
[N--D---] FOP.MUL R87, R184.l, R87
[N--D---] FOP.MOV R76.h.bf16, R87.f32
[N--D---] FOP.MUL R95, R184.l, R95
[N-sD---] FOP.MOV R77.h.bf16, R95.f32
[NswD---]       DMA.ST.B64 R[76:77], SH[0:1], R167, _, _, _, _,  SH[0:2]
[NwwDwww]    CTRL.NOP
[NwwD---]    AP.BARRIER LM0
[NwwD---]    CTRL.END
STORE_MASK_RELU:
[NwwD---] FOP.FMA R32, R32, R185.l, R144
[N--D---] FOP.FMA R33, R33, R185.h, R145
[N--D---] FOP.FMA R40, R40, R186.l, R146
[N-sD---] FOP.FMA R41, R41, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R32, R32, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R33, R33, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R40, R40, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R41, R41, C0
[NwwD---] FOP.MUL R32, R32, R169.l
[N--D---] FOP.MOV R32.l.bf16, R32.f32
[N--D---] FOP.MUL R33, R33, R169.l
[N--D---] FOP.MOV R33.l.bf16, R33.f32
[N--D---] FOP.MUL R40, R40, R169.l
[N--D---] FOP.MOV R40.l.bf16, R40.f32
[N--D---] FOP.MUL R41, R41, R169.l
[N-sD---] FOP.MOV R41.l.bf16, R41.f32
[NswD---]       INT.BYP.TG p0, R3.s32, C0.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R32.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R33.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R40.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R41.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NwwD---] FOP.FMA R34, R34, R185.l, R144
[N--D---] FOP.FMA R35, R35, R185.h, R145
[N--D---] FOP.FMA R42, R42, R186.l, R146
[N-sD---] FOP.FMA R43, R43, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R34, R34, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R35, R35, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R42, R42, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R43, R43, C0
[NwwD---] FOP.MUL R34, R34, R170.l
[N--D---] FOP.MOV R34.l.bf16, R34.f32
[N--D---] FOP.MUL R35, R35, R170.l
[N--D---] FOP.MOV R35.l.bf16, R35.f32
[N--D---] FOP.MUL R42, R42, R170.l
[N--D---] FOP.MOV R42.l.bf16, R42.f32
[N--D---] FOP.MUL R43, R43, R170.l
[N-sD---] FOP.MOV R43.l.bf16, R43.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, C46.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R34.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R35.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R42.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R43.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NwwD---] FOP.FMA R36, R36, R185.l, R144
[N--D---] FOP.FMA R37, R37, R185.h, R145
[N--D---] FOP.FMA R44, R44, R186.l, R146
[N-sD---] FOP.FMA R45, R45, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R36, R36, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R37, R37, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R44, R44, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R45, R45, C0
[NwwD---] FOP.MUL R36, R36, R171.l
[N--D---] FOP.MOV R36.l.bf16, R36.f32
[N--D---] FOP.MUL R37, R37, R171.l
[N--D---] FOP.MOV R37.l.bf16, R37.f32
[N--D---] FOP.MUL R44, R44, R171.l
[N--D---] FOP.MOV R44.l.bf16, R44.f32
[N--D---] FOP.MUL R45, R45, R171.l
[N-sD---] FOP.MOV R45.l.bf16, R45.f32
[NswD---]       INT.BYP.TG p0, R3.s32, C39.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R36.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R37.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R44.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R45.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NwwD---] FOP.FMA R38, R38, R185.l, R144
[N--D---] FOP.FMA R39, R39, R185.h, R145
[N--D---] FOP.FMA R46, R46, R186.l, R146
[N-sD---] FOP.FMA R47, R47, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R38, R38, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R39, R39, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R46, R46, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R47, R47, C0
[NwwD---] FOP.MUL R38, R38, R172.l
[N--D---] FOP.MOV R38.l.bf16, R38.f32
[N--D---] FOP.MUL R39, R39, R172.l
[N--D---] FOP.MOV R39.l.bf16, R39.f32
[N--D---] FOP.MUL R46, R46, R172.l
[N--D---] FOP.MOV R46.l.bf16, R46.f32
[N--D---] FOP.MUL R47, R47, R172.l
[N-sD---] FOP.MOV R47.l.bf16, R47.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, C48.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R38.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R39.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R46.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R47.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NwwD---] FOP.FMA R48, R48, R185.l, R144
[N--D---] FOP.FMA R49, R49, R185.h, R145
[N--D---] FOP.FMA R56, R56, R186.l, R146
[N-sD---] FOP.FMA R57, R57, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R48, R48, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R49, R49, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R56, R56, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R57, R57, C0
[NwwD---] FOP.MUL R48, R48, R173.l
[N--D---] FOP.MOV R48.l.bf16, R48.f32
[N--D---] FOP.MUL R49, R49, R173.l
[N--D---] FOP.MOV R49.l.bf16, R49.f32
[N--D---] FOP.MUL R56, R56, R173.l
[N--D---] FOP.MOV R56.l.bf16, R56.f32
[N--D---] FOP.MUL R57, R57, R173.l
[N-sD---] FOP.MOV R57.l.bf16, R57.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH90.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R48.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R49.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R56.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R57.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NwwD---] FOP.FMA R50, R50, R185.l, R144
[N--D---] FOP.FMA R51, R51, R185.h, R145
[N--D---] FOP.FMA R58, R58, R186.l, R146
[N-sD---] FOP.FMA R59, R59, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R50, R50, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R51, R51, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R58, R58, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R59, R59, C0
[NwwD---] FOP.MUL R50, R50, R174.l
[N--D---] FOP.MOV R50.l.bf16, R50.f32
[N--D---] FOP.MUL R51, R51, R174.l
[N--D---] FOP.MOV R51.l.bf16, R51.f32
[N--D---] FOP.MUL R58, R58, R174.l
[N--D---] FOP.MOV R58.l.bf16, R58.f32
[N--D---] FOP.MUL R59, R59, R174.l
[N-sD---] FOP.MOV R59.l.bf16, R59.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH91.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R50.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R51.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R58.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R59.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NwwD---] FOP.FMA R52, R52, R185.l, R144
[N--D---] FOP.FMA R53, R53, R185.h, R145
[N--D---] FOP.FMA R60, R60, R186.l, R146
[N-sD---] FOP.FMA R61, R61, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R52, R52, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R53, R53, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R60, R60, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R61, R61, C0
[NwwD---] FOP.MUL R52, R52, R175.l
[N--D---] FOP.MOV R52.l.bf16, R52.f32
[N--D---] FOP.MUL R53, R53, R175.l
[N--D---] FOP.MOV R53.l.bf16, R53.f32
[N--D---] FOP.MUL R60, R60, R175.l
[N--D---] FOP.MOV R60.l.bf16, R60.f32
[N--D---] FOP.MUL R61, R61, R175.l
[N-sD---] FOP.MOV R61.l.bf16, R61.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH92.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R52.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R53.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R60.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R61.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NwwD---] FOP.FMA R54, R54, R185.l, R144
[N--D---] FOP.FMA R55, R55, R185.h, R145
[N--D---] FOP.FMA R62, R62, R186.l, R146
[N-sD---] FOP.FMA R63, R63, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R54, R54, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R55, R55, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R62, R62, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R63, R63, C0
[NwwD---] FOP.MUL R54, R54, R176.l
[N--D---] FOP.MOV R54.l.bf16, R54.f32
[N--D---] FOP.MUL R55, R55, R176.l
[N--D---] FOP.MOV R55.l.bf16, R55.f32
[N--D---] FOP.MUL R62, R62, R176.l
[N--D---] FOP.MOV R62.l.bf16, R62.f32
[N--D---] FOP.MUL R63, R63, R176.l
[N-sD---] FOP.MOV R63.l.bf16, R63.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH93.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R54.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R55.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R62.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R63.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NwwD---] FOP.FMA R64, R64, R185.l, R144
[N--D---] FOP.FMA R65, R65, R185.h, R145
[N--D---] FOP.FMA R72, R72, R186.l, R146
[N-sD---] FOP.FMA R73, R73, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R64, R64, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R65, R65, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R72, R72, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R73, R73, C0
[NwwD---] FOP.MUL R64, R64, R177.l
[N--D---] FOP.MOV R64.l.bf16, R64.f32
[N--D---] FOP.MUL R65, R65, R177.l
[N--D---] FOP.MOV R65.l.bf16, R65.f32
[N--D---] FOP.MUL R72, R72, R177.l
[N--D---] FOP.MOV R72.l.bf16, R72.f32
[N--D---] FOP.MUL R73, R73, R177.l
[N-sD---] FOP.MOV R73.l.bf16, R73.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH95.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R64.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R65.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R72.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R73.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NwwD---] FOP.FMA R66, R66, R185.l, R144
[N--D---] FOP.FMA R67, R67, R185.h, R145
[N--D---] FOP.FMA R74, R74, R186.l, R146
[N-sD---] FOP.FMA R75, R75, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R66, R66, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R67, R67, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R74, R74, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R75, R75, C0
[NwwD---] FOP.MUL R66, R66, R178.l
[N--D---] FOP.MOV R66.l.bf16, R66.f32
[N--D---] FOP.MUL R67, R67, R178.l
[N--D---] FOP.MOV R67.l.bf16, R67.f32
[N--D---] FOP.MUL R74, R74, R178.l
[N--D---] FOP.MOV R74.l.bf16, R74.f32
[N--D---] FOP.MUL R75, R75, R178.l
[N-sD---] FOP.MOV R75.l.bf16, R75.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH96.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R66.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R67.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R74.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R75.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NwwD---] FOP.FMA R68, R68, R185.l, R144
[N--D---] FOP.FMA R69, R69, R185.h, R145
[N--D---] FOP.FMA R76, R76, R186.l, R146
[N-sD---] FOP.FMA R77, R77, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R68, R68, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R69, R69, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R76, R76, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R77, R77, C0
[NwwD---] FOP.MUL R68, R68, R179.l
[N--D---] FOP.MOV R68.l.bf16, R68.f32
[N--D---] FOP.MUL R69, R69, R179.l
[N--D---] FOP.MOV R69.l.bf16, R69.f32
[N--D---] FOP.MUL R76, R76, R179.l
[N--D---] FOP.MOV R76.l.bf16, R76.f32
[N--D---] FOP.MUL R77, R77, R179.l
[N-sD---] FOP.MOV R77.l.bf16, R77.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH97.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R68.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R69.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R76.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R77.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NwwD---] FOP.FMA R70, R70, R185.l, R144
[N--D---] FOP.FMA R71, R71, R185.h, R145
[N--D---] FOP.FMA R78, R78, R186.l, R146
[N-sD---] FOP.FMA R79, R79, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R70, R70, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R71, R71, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R78, R78, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R79, R79, C0
[NwwD---] FOP.MUL R70, R70, R180.l
[N--D---] FOP.MOV R70.l.bf16, R70.f32
[N--D---] FOP.MUL R71, R71, R180.l
[N--D---] FOP.MOV R71.l.bf16, R71.f32
[N--D---] FOP.MUL R78, R78, R180.l
[N--D---] FOP.MOV R78.l.bf16, R78.f32
[N--D---] FOP.MUL R79, R79, R180.l
[N-sD---] FOP.MOV R79.l.bf16, R79.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH98.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R70.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R71.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R78.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R79.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NwwD---] FOP.FMA R80, R80, R185.l, R144
[N--D---] FOP.FMA R81, R81, R185.h, R145
[N--D---] FOP.FMA R88, R88, R186.l, R146
[N-sD---] FOP.FMA R89, R89, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R80, R80, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R81, R81, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R88, R88, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R89, R89, C0
[NwwD---] FOP.MUL R80, R80, R181.l
[N--D---] FOP.MOV R80.l.bf16, R80.f32
[N--D---] FOP.MUL R81, R81, R181.l
[N--D---] FOP.MOV R81.l.bf16, R81.f32
[N--D---] FOP.MUL R88, R88, R181.l
[N--D---] FOP.MOV R88.l.bf16, R88.f32
[N--D---] FOP.MUL R89, R89, R181.l
[N-sD---] FOP.MOV R89.l.bf16, R89.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH99.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R80.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R81.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R88.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R89.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NwwD---] FOP.FMA R82, R82, R185.l, R144
[N--D---] FOP.FMA R83, R83, R185.h, R145
[N--D---] FOP.FMA R90, R90, R186.l, R146
[N-sD---] FOP.FMA R91, R91, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R82, R82, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R83, R83, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R90, R90, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R91, R91, C0
[NwwD---] FOP.MUL R82, R82, R182.l
[N--D---] FOP.MOV R82.l.bf16, R82.f32
[N--D---] FOP.MUL R83, R83, R182.l
[N--D---] FOP.MOV R83.l.bf16, R83.f32
[N--D---] FOP.MUL R90, R90, R182.l
[N--D---] FOP.MOV R90.l.bf16, R90.f32
[N--D---] FOP.MUL R91, R91, R182.l
[N-sD---] FOP.MOV R91.l.bf16, R91.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH100.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R82.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R83.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R90.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R91.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NwwD---] FOP.FMA R84, R84, R185.l, R144
[N--D---] FOP.FMA R85, R85, R185.h, R145
[N--D---] FOP.FMA R92, R92, R186.l, R146
[N-sD---] FOP.FMA R93, R93, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R84, R84, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R85, R85, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R92, R92, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R93, R93, C0
[NwwD---] FOP.MUL R84, R84, R183.l
[N--D---] FOP.MOV R84.l.bf16, R84.f32
[N--D---] FOP.MUL R85, R85, R183.l
[N--D---] FOP.MOV R85.l.bf16, R85.f32
[N--D---] FOP.MUL R92, R92, R183.l
[N--D---] FOP.MOV R92.l.bf16, R92.f32
[N--D---] FOP.MUL R93, R93, R183.l
[N-sD---] FOP.MOV R93.l.bf16, R93.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH101.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R84.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R85.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R92.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R93.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NwwD---] FOP.FMA R86, R86, R185.l, R144
[N--D---] FOP.FMA R87, R87, R185.h, R145
[N--D---] FOP.FMA R94, R94, R186.l, R146
[N-sD---] FOP.FMA R95, R95, R186.h, R147
[N--D---]    FOP.CMP_MOVC.MAX.TST R86, R86, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R87, R87, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R94, R94, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R95, R95, C0
[NwwD---] FOP.MUL R86, R86, R184.l
[N--D---] FOP.MOV R86.l.bf16, R86.f32
[N--D---] FOP.MUL R87, R87, R184.l
[N--D---] FOP.MOV R87.l.bf16, R87.f32
[N--D---] FOP.MUL R94, R94, R184.l
[N--D---] FOP.MOV R94.l.bf16, R94.f32
[N--D---] FOP.MUL R95, R95, R184.l
[N-sD---] FOP.MOV R95.l.bf16, R95.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH102.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R86.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R87.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R94.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R95.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NwwDwww]    CTRL.NOP
[NwwD---]    AP.BARRIER LM0
[NwwD---]    CTRL.END
LEAKYRELU_START:
[NswD---]       INT.BYP.TLE p0, R3.s32, SH102.s32
[NwsD---]       INT.BYP.TLE.OP p0, R4.s32, C59.s32
[NwwD---]        CTRL.BR.ANYP0T STORE_MASK_LEAKYRELU
[N-wD---] FOP.FMA R32, R32, R185.l, R144
[N--D---] FOP.FMA R33, R33, R185.h, R145
[N--D---] FOP.FMA R40, R40, R186.l, R146
[N-sD---] FOP.FMA R41, R41, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R32, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R32, R32, R168
[N--D---]   FOP.CMP.LT p0, R33, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R33, R33, R168
[N--D---]   FOP.CMP.LT p0, R40, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R40, R40, R168
[N--D---]   FOP.CMP.LT p0, R41, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R41, R41, R168
[N--D---] FOP.MUL R32, R169.l, R32
[N--D---] FOP.MOV R112.l.bf16, R32.f32
[N--D---] FOP.MUL R40, R169.l, R40
[N--D---] FOP.MOV R113.l.bf16, R40.f32
[N--D---] FOP.MUL R33, R169.l, R33
[N--D---] FOP.MOV R112.h.bf16, R33.f32
[N--D---] FOP.MUL R41, R169.l, R41
[N-sD---] FOP.MOV R113.h.bf16, R41.f32
[NwwD---]       DMA.ST.B64 R[112:113], SH[0:1], R152, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R34, R34, R185.l, R144
[N--D---] FOP.FMA R35, R35, R185.h, R145
[N--D---] FOP.FMA R42, R42, R186.l, R146
[N-sD---] FOP.FMA R43, R43, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R34, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R34, R34, R168
[N--D---]   FOP.CMP.LT p0, R35, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R35, R35, R168
[N--D---]   FOP.CMP.LT p0, R42, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R42, R42, R168
[N--D---]   FOP.CMP.LT p0, R43, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R43, R43, R168
[N--D---] FOP.MUL R34, R170.l, R34
[N--D---] FOP.MOV R114.l.bf16, R34.f32
[N--D---] FOP.MUL R42, R170.l, R42
[N--D---] FOP.MOV R115.l.bf16, R42.f32
[N--D---] FOP.MUL R35, R170.l, R35
[N--D---] FOP.MOV R114.h.bf16, R35.f32
[N--D---] FOP.MUL R43, R170.l, R43
[N-sD---] FOP.MOV R115.h.bf16, R43.f32
[NwwD---]       DMA.ST.B64 R[114:115], SH[0:1], R153, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R36, R36, R185.l, R144
[N--D---] FOP.FMA R37, R37, R185.h, R145
[N--D---] FOP.FMA R44, R44, R186.l, R146
[N-sD---] FOP.FMA R45, R45, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R36, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R36, R36, R168
[N--D---]   FOP.CMP.LT p0, R37, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R37, R37, R168
[N--D---]   FOP.CMP.LT p0, R44, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R44, R44, R168
[N--D---]   FOP.CMP.LT p0, R45, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R45, R45, R168
[N--D---] FOP.MUL R36, R171.l, R36
[N--D---] FOP.MOV R116.l.bf16, R36.f32
[N--D---] FOP.MUL R44, R171.l, R44
[N--D---] FOP.MOV R117.l.bf16, R44.f32
[N--D---] FOP.MUL R37, R171.l, R37
[N--D---] FOP.MOV R116.h.bf16, R37.f32
[N--D---] FOP.MUL R45, R171.l, R45
[N-sD---] FOP.MOV R117.h.bf16, R45.f32
[NwwD---]       DMA.ST.B64 R[116:117], SH[0:1], R154, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R38, R38, R185.l, R144
[N--D---] FOP.FMA R39, R39, R185.h, R145
[N--D---] FOP.FMA R46, R46, R186.l, R146
[N-sD---] FOP.FMA R47, R47, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R38, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R38, R38, R168
[N--D---]   FOP.CMP.LT p0, R39, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R39, R39, R168
[N--D---]   FOP.CMP.LT p0, R46, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R46, R46, R168
[N--D---]   FOP.CMP.LT p0, R47, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R47, R47, R168
[N--D---] FOP.MUL R38, R172.l, R38
[N--D---] FOP.MOV R118.l.bf16, R38.f32
[N--D---] FOP.MUL R46, R172.l, R46
[N--D---] FOP.MOV R119.l.bf16, R46.f32
[N--D---] FOP.MUL R39, R172.l, R39
[N--D---] FOP.MOV R118.h.bf16, R39.f32
[N--D---] FOP.MUL R47, R172.l, R47
[N-sD---] FOP.MOV R119.h.bf16, R47.f32
[NwwD---]       DMA.ST.B64 R[118:119], SH[0:1], R155, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R48, R48, R185.l, R144
[N--D---] FOP.FMA R49, R49, R185.h, R145
[N--D---] FOP.FMA R56, R56, R186.l, R146
[N-sD---] FOP.FMA R57, R57, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R48, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R48, R48, R168
[N--D---]   FOP.CMP.LT p0, R49, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R49, R49, R168
[N--D---]   FOP.CMP.LT p0, R56, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R56, R56, R168
[N--D---]   FOP.CMP.LT p0, R57, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R57, R57, R168
[N--D---] FOP.MUL R48, R173.l, R48
[N--D---] FOP.MOV R32.l.bf16, R48.f32
[N--D---] FOP.MUL R56, R173.l, R56
[N--D---] FOP.MOV R33.l.bf16, R56.f32
[N--D---] FOP.MUL R49, R173.l, R49
[N--D---] FOP.MOV R32.h.bf16, R49.f32
[N--D---] FOP.MUL R57, R173.l, R57
[N-sD---] FOP.MOV R33.h.bf16, R57.f32
[NwwD---]       DMA.ST.B64 R[32:33], SH[0:1], R156, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R50, R50, R185.l, R144
[N--D---] FOP.FMA R51, R51, R185.h, R145
[N--D---] FOP.FMA R58, R58, R186.l, R146
[N-sD---] FOP.FMA R59, R59, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R50, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R50, R50, R168
[N--D---]   FOP.CMP.LT p0, R51, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R51, R51, R168
[N--D---]   FOP.CMP.LT p0, R58, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R58, R58, R168
[N--D---]   FOP.CMP.LT p0, R59, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R59, R59, R168
[N--D---] FOP.MUL R50, R174.l, R50
[N--D---] FOP.MOV R36.l.bf16, R50.f32
[N--D---] FOP.MUL R58, R174.l, R58
[N--D---] FOP.MOV R37.l.bf16, R58.f32
[N--D---] FOP.MUL R51, R174.l, R51
[N--D---] FOP.MOV R36.h.bf16, R51.f32
[N--D---] FOP.MUL R59, R174.l, R59
[N-sD---] FOP.MOV R37.h.bf16, R59.f32
[NwwD---]       DMA.ST.B64 R[36:37], SH[0:1], R157, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R52, R52, R185.l, R144
[N--D---] FOP.FMA R53, R53, R185.h, R145
[N--D---] FOP.FMA R60, R60, R186.l, R146
[N-sD---] FOP.FMA R61, R61, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R52, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R52, R52, R168
[N--D---]   FOP.CMP.LT p0, R53, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R53, R53, R168
[N--D---]   FOP.CMP.LT p0, R60, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R60, R60, R168
[N--D---]   FOP.CMP.LT p0, R61, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R61, R61, R168
[N--D---] FOP.MUL R52, R175.l, R52
[N--D---] FOP.MOV R40.l.bf16, R52.f32
[N--D---] FOP.MUL R60, R175.l, R60
[N--D---] FOP.MOV R41.l.bf16, R60.f32
[N--D---] FOP.MUL R53, R175.l, R53
[N--D---] FOP.MOV R40.h.bf16, R53.f32
[N--D---] FOP.MUL R61, R175.l, R61
[N-sD---] FOP.MOV R41.h.bf16, R61.f32
[NwwD---]       DMA.ST.B64 R[40:41], SH[0:1], R158, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R54, R54, R185.l, R144
[N--D---] FOP.FMA R55, R55, R185.h, R145
[N--D---] FOP.FMA R62, R62, R186.l, R146
[N-sD---] FOP.FMA R63, R63, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R54, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R54, R54, R168
[N--D---]   FOP.CMP.LT p0, R55, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R55, R55, R168
[N--D---]   FOP.CMP.LT p0, R62, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R62, R62, R168
[N--D---]   FOP.CMP.LT p0, R63, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R63, R63, R168
[N--D---] FOP.MUL R54, R176.l, R54
[N--D---] FOP.MOV R44.l.bf16, R54.f32
[N--D---] FOP.MUL R62, R176.l, R62
[N--D---] FOP.MOV R45.l.bf16, R62.f32
[N--D---] FOP.MUL R55, R176.l, R55
[N--D---] FOP.MOV R44.h.bf16, R55.f32
[N--D---] FOP.MUL R63, R176.l, R63
[N-sD---] FOP.MOV R45.h.bf16, R63.f32
[NwwD---]       DMA.ST.B64 R[44:45], SH[0:1], R159, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R64, R64, R185.l, R144
[N--D---] FOP.FMA R65, R65, R185.h, R145
[N--D---] FOP.FMA R72, R72, R186.l, R146
[N-sD---] FOP.FMA R73, R73, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R64, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R64, R64, R168
[N--D---]   FOP.CMP.LT p0, R65, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R65, R65, R168
[N--D---]   FOP.CMP.LT p0, R72, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R72, R72, R168
[N--D---]   FOP.CMP.LT p0, R73, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R73, R73, R168
[N--D---] FOP.MUL R64, R177.l, R64
[N--D---] FOP.MOV R48.l.bf16, R64.f32
[N--D---] FOP.MUL R72, R177.l, R72
[N--D---] FOP.MOV R49.l.bf16, R72.f32
[N--D---] FOP.MUL R65, R177.l, R65
[N--D---] FOP.MOV R48.h.bf16, R65.f32
[N--D---] FOP.MUL R73, R177.l, R73
[N-sD---] FOP.MOV R49.h.bf16, R73.f32
[NwwD---]       DMA.ST.B64 R[48:49], SH[0:1], R160, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R66, R66, R185.l, R144
[N--D---] FOP.FMA R67, R67, R185.h, R145
[N--D---] FOP.FMA R74, R74, R186.l, R146
[N-sD---] FOP.FMA R75, R75, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R66, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R66, R66, R168
[N--D---]   FOP.CMP.LT p0, R67, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R67, R67, R168
[N--D---]   FOP.CMP.LT p0, R74, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R74, R74, R168
[N--D---]   FOP.CMP.LT p0, R75, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R75, R75, R168
[N--D---] FOP.MUL R66, R178.l, R66
[N--D---] FOP.MOV R52.l.bf16, R66.f32
[N--D---] FOP.MUL R74, R178.l, R74
[N--D---] FOP.MOV R53.l.bf16, R74.f32
[N--D---] FOP.MUL R67, R178.l, R67
[N--D---] FOP.MOV R52.h.bf16, R67.f32
[N--D---] FOP.MUL R75, R178.l, R75
[N-sD---] FOP.MOV R53.h.bf16, R75.f32
[NwwD---]       DMA.ST.B64 R[52:53], SH[0:1], R161, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R68, R68, R185.l, R144
[N--D---] FOP.FMA R69, R69, R185.h, R145
[N--D---] FOP.FMA R76, R76, R186.l, R146
[N-sD---] FOP.FMA R77, R77, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R68, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R68, R68, R168
[N--D---]   FOP.CMP.LT p0, R69, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R69, R69, R168
[N--D---]   FOP.CMP.LT p0, R76, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R76, R76, R168
[N--D---]   FOP.CMP.LT p0, R77, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R77, R77, R168
[N--D---] FOP.MUL R68, R179.l, R68
[N--D---] FOP.MOV R56.l.bf16, R68.f32
[N--D---] FOP.MUL R76, R179.l, R76
[N--D---] FOP.MOV R57.l.bf16, R76.f32
[N--D---] FOP.MUL R69, R179.l, R69
[N--D---] FOP.MOV R56.h.bf16, R69.f32
[N--D---] FOP.MUL R77, R179.l, R77
[N-sD---] FOP.MOV R57.h.bf16, R77.f32
[NwwD---]       DMA.ST.B64 R[56:57], SH[0:1], R162, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R70, R70, R185.l, R144
[N--D---] FOP.FMA R71, R71, R185.h, R145
[N--D---] FOP.FMA R78, R78, R186.l, R146
[N-sD---] FOP.FMA R79, R79, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R70, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R70, R70, R168
[N--D---]   FOP.CMP.LT p0, R71, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R71, R71, R168
[N--D---]   FOP.CMP.LT p0, R78, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R78, R78, R168
[N--D---]   FOP.CMP.LT p0, R79, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R79, R79, R168
[N--D---] FOP.MUL R70, R180.l, R70
[N--D---] FOP.MOV R60.l.bf16, R70.f32
[N--D---] FOP.MUL R78, R180.l, R78
[N--D---] FOP.MOV R61.l.bf16, R78.f32
[N--D---] FOP.MUL R71, R180.l, R71
[N--D---] FOP.MOV R60.h.bf16, R71.f32
[N--D---] FOP.MUL R79, R180.l, R79
[N-sD---] FOP.MOV R61.h.bf16, R79.f32
[NwwD---]       DMA.ST.B64 R[60:61], SH[0:1], R163, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R80, R80, R185.l, R144
[N--D---] FOP.FMA R81, R81, R185.h, R145
[N--D---] FOP.FMA R88, R88, R186.l, R146
[N-sD---] FOP.FMA R89, R89, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R80, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R80, R80, R168
[N--D---]   FOP.CMP.LT p0, R81, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R81, R81, R168
[N--D---]   FOP.CMP.LT p0, R88, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R88, R88, R168
[N--D---]   FOP.CMP.LT p0, R89, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R89, R89, R168
[N--D---] FOP.MUL R80, R181.l, R80
[N--D---] FOP.MOV R64.l.bf16, R80.f32
[N--D---] FOP.MUL R88, R181.l, R88
[N--D---] FOP.MOV R65.l.bf16, R88.f32
[N--D---] FOP.MUL R81, R181.l, R81
[N--D---] FOP.MOV R64.h.bf16, R81.f32
[N--D---] FOP.MUL R89, R181.l, R89
[N-sD---] FOP.MOV R65.h.bf16, R89.f32
[NwwD---]       DMA.ST.B64 R[64:65], SH[0:1], R164, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R82, R82, R185.l, R144
[N--D---] FOP.FMA R83, R83, R185.h, R145
[N--D---] FOP.FMA R90, R90, R186.l, R146
[N-sD---] FOP.FMA R91, R91, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R82, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R82, R82, R168
[N--D---]   FOP.CMP.LT p0, R83, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R83, R83, R168
[N--D---]   FOP.CMP.LT p0, R90, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R90, R90, R168
[N--D---]   FOP.CMP.LT p0, R91, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R91, R91, R168
[N--D---] FOP.MUL R82, R182.l, R82
[N--D---] FOP.MOV R68.l.bf16, R82.f32
[N--D---] FOP.MUL R90, R182.l, R90
[N--D---] FOP.MOV R69.l.bf16, R90.f32
[N--D---] FOP.MUL R83, R182.l, R83
[N--D---] FOP.MOV R68.h.bf16, R83.f32
[N--D---] FOP.MUL R91, R182.l, R91
[N-sD---] FOP.MOV R69.h.bf16, R91.f32
[NwwD---]       DMA.ST.B64 R[68:69], SH[0:1], R165, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R84, R84, R185.l, R144
[N--D---] FOP.FMA R85, R85, R185.h, R145
[N--D---] FOP.FMA R92, R92, R186.l, R146
[N-sD---] FOP.FMA R93, R93, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R84, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R84, R84, R168
[N--D---]   FOP.CMP.LT p0, R85, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R85, R85, R168
[N--D---]   FOP.CMP.LT p0, R92, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R92, R92, R168
[N--D---]   FOP.CMP.LT p0, R93, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R93, R93, R168
[N--D---] FOP.MUL R84, R183.l, R84
[N--D---] FOP.MOV R72.l.bf16, R84.f32
[N--D---] FOP.MUL R92, R183.l, R92
[N--D---] FOP.MOV R73.l.bf16, R92.f32
[N--D---] FOP.MUL R85, R183.l, R85
[N--D---] FOP.MOV R72.h.bf16, R85.f32
[N--D---] FOP.MUL R93, R183.l, R93
[N-sD---] FOP.MOV R73.h.bf16, R93.f32
[NwwD---]       DMA.ST.B64 R[72:73], SH[0:1], R166, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R86, R86, R185.l, R144
[N--D---] FOP.FMA R87, R87, R185.h, R145
[N--D---] FOP.FMA R94, R94, R186.l, R146
[N-sD---] FOP.FMA R95, R95, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R86, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R86, R86, R168
[N--D---]   FOP.CMP.LT p0, R87, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R87, R87, R168
[N--D---]   FOP.CMP.LT p0, R94, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R94, R94, R168
[N--D---]   FOP.CMP.LT p0, R95, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R95, R95, R168
[N--D---] FOP.MUL R86, R184.l, R86
[N--D---] FOP.MOV R76.l.bf16, R86.f32
[N--D---] FOP.MUL R94, R184.l, R94
[N--D---] FOP.MOV R77.l.bf16, R94.f32
[N--D---] FOP.MUL R87, R184.l, R87
[N--D---] FOP.MOV R76.h.bf16, R87.f32
[N--D---] FOP.MUL R95, R184.l, R95
[N-sD---] FOP.MOV R77.h.bf16, R95.f32
[NswD---]       DMA.ST.B64 R[76:77], SH[0:1], R167, _, _, _, _,  SH[0:2]
[NwwDwww]    CTRL.NOP
[NwwD---]    AP.BARRIER LM0
[NwwD---]    CTRL.END
STORE_MASK_LEAKYRELU:
[NwwD---] FOP.FMA R32, R32, R185.l, R144
[N--D---] FOP.FMA R33, R33, R185.h, R145
[N--D---] FOP.FMA R40, R40, R186.l, R146
[N-sD---] FOP.FMA R41, R41, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R32, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R32, R32, R168
[N--D---]   FOP.CMP.LT p0, R33, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R33, R33, R168
[N--D---]   FOP.CMP.LT p0, R40, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R40, R40, R168
[N--D---]   FOP.CMP.LT p0, R41, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R41, R41, R168
[NwwD---] FOP.MUL R32, R32, R169.l
[N--D---] FOP.MOV R32.l.bf16, R32.f32
[N--D---] FOP.MUL R33, R33, R169.l
[N--D---] FOP.MOV R33.l.bf16, R33.f32
[N--D---] FOP.MUL R40, R40, R169.l
[N--D---] FOP.MOV R40.l.bf16, R40.f32
[N--D---] FOP.MUL R41, R41, R169.l
[N-sD---] FOP.MOV R41.l.bf16, R41.f32
[NswD---]       INT.BYP.TG p0, R3.s32, C0.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R32.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R33.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R40.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R41.l, SH[0:1], R152, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R152.s32, C45.s32, R152.s32
[NwwD---] FOP.FMA R34, R34, R185.l, R144
[N--D---] FOP.FMA R35, R35, R185.h, R145
[N--D---] FOP.FMA R42, R42, R186.l, R146
[N-sD---] FOP.FMA R43, R43, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R34, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R34, R34, R168
[N--D---]   FOP.CMP.LT p0, R35, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R35, R35, R168
[N--D---]   FOP.CMP.LT p0, R42, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R42, R42, R168
[N--D---]   FOP.CMP.LT p0, R43, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R43, R43, R168
[NwwD---] FOP.MUL R34, R34, R170.l
[N--D---] FOP.MOV R34.l.bf16, R34.f32
[N--D---] FOP.MUL R35, R35, R170.l
[N--D---] FOP.MOV R35.l.bf16, R35.f32
[N--D---] FOP.MUL R42, R42, R170.l
[N--D---] FOP.MOV R42.l.bf16, R42.f32
[N--D---] FOP.MUL R43, R43, R170.l
[N-sD---] FOP.MOV R43.l.bf16, R43.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, C46.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R34.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R35.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R42.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R43.l, SH[0:1], R153, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R153.s32, C45.s32, R153.s32
[NwwD---] FOP.FMA R36, R36, R185.l, R144
[N--D---] FOP.FMA R37, R37, R185.h, R145
[N--D---] FOP.FMA R44, R44, R186.l, R146
[N-sD---] FOP.FMA R45, R45, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R36, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R36, R36, R168
[N--D---]   FOP.CMP.LT p0, R37, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R37, R37, R168
[N--D---]   FOP.CMP.LT p0, R44, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R44, R44, R168
[N--D---]   FOP.CMP.LT p0, R45, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R45, R45, R168
[NwwD---] FOP.MUL R36, R36, R171.l
[N--D---] FOP.MOV R36.l.bf16, R36.f32
[N--D---] FOP.MUL R37, R37, R171.l
[N--D---] FOP.MOV R37.l.bf16, R37.f32
[N--D---] FOP.MUL R44, R44, R171.l
[N--D---] FOP.MOV R44.l.bf16, R44.f32
[N--D---] FOP.MUL R45, R45, R171.l
[N-sD---] FOP.MOV R45.l.bf16, R45.f32
[NswD---]       INT.BYP.TG p0, R3.s32, C39.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R36.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R37.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R44.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R45.l, SH[0:1], R154, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R154.s32, C45.s32, R154.s32
[NwwD---] FOP.FMA R38, R38, R185.l, R144
[N--D---] FOP.FMA R39, R39, R185.h, R145
[N--D---] FOP.FMA R46, R46, R186.l, R146
[N-sD---] FOP.FMA R47, R47, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R38, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R38, R38, R168
[N--D---]   FOP.CMP.LT p0, R39, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R39, R39, R168
[N--D---]   FOP.CMP.LT p0, R46, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R46, R46, R168
[N--D---]   FOP.CMP.LT p0, R47, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R47, R47, R168
[NwwD---] FOP.MUL R38, R38, R172.l
[N--D---] FOP.MOV R38.l.bf16, R38.f32
[N--D---] FOP.MUL R39, R39, R172.l
[N--D---] FOP.MOV R39.l.bf16, R39.f32
[N--D---] FOP.MUL R46, R46, R172.l
[N--D---] FOP.MOV R46.l.bf16, R46.f32
[N--D---] FOP.MUL R47, R47, R172.l
[N-sD---] FOP.MOV R47.l.bf16, R47.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, C48.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R38.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R39.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R46.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R47.l, SH[0:1], R155, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R155.s32, C45.s32, R155.s32
[NwwD---] FOP.FMA R48, R48, R185.l, R144
[N--D---] FOP.FMA R49, R49, R185.h, R145
[N--D---] FOP.FMA R56, R56, R186.l, R146
[N-sD---] FOP.FMA R57, R57, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R48, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R48, R48, R168
[N--D---]   FOP.CMP.LT p0, R49, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R49, R49, R168
[N--D---]   FOP.CMP.LT p0, R56, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R56, R56, R168
[N--D---]   FOP.CMP.LT p0, R57, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R57, R57, R168
[NwwD---] FOP.MUL R48, R48, R173.l
[N--D---] FOP.MOV R48.l.bf16, R48.f32
[N--D---] FOP.MUL R49, R49, R173.l
[N--D---] FOP.MOV R49.l.bf16, R49.f32
[N--D---] FOP.MUL R56, R56, R173.l
[N--D---] FOP.MOV R56.l.bf16, R56.f32
[N--D---] FOP.MUL R57, R57, R173.l
[N-sD---] FOP.MOV R57.l.bf16, R57.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH90.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R48.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R49.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R56.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R57.l, SH[0:1], R156, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R156.s32, C45.s32, R156.s32
[NwwD---] FOP.FMA R50, R50, R185.l, R144
[N--D---] FOP.FMA R51, R51, R185.h, R145
[N--D---] FOP.FMA R58, R58, R186.l, R146
[N-sD---] FOP.FMA R59, R59, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R50, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R50, R50, R168
[N--D---]   FOP.CMP.LT p0, R51, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R51, R51, R168
[N--D---]   FOP.CMP.LT p0, R58, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R58, R58, R168
[N--D---]   FOP.CMP.LT p0, R59, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R59, R59, R168
[NwwD---] FOP.MUL R50, R50, R174.l
[N--D---] FOP.MOV R50.l.bf16, R50.f32
[N--D---] FOP.MUL R51, R51, R174.l
[N--D---] FOP.MOV R51.l.bf16, R51.f32
[N--D---] FOP.MUL R58, R58, R174.l
[N--D---] FOP.MOV R58.l.bf16, R58.f32
[N--D---] FOP.MUL R59, R59, R174.l
[N-sD---] FOP.MOV R59.l.bf16, R59.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH91.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R50.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R51.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R58.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R59.l, SH[0:1], R157, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R157.s32, C45.s32, R157.s32
[NwwD---] FOP.FMA R52, R52, R185.l, R144
[N--D---] FOP.FMA R53, R53, R185.h, R145
[N--D---] FOP.FMA R60, R60, R186.l, R146
[N-sD---] FOP.FMA R61, R61, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R52, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R52, R52, R168
[N--D---]   FOP.CMP.LT p0, R53, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R53, R53, R168
[N--D---]   FOP.CMP.LT p0, R60, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R60, R60, R168
[N--D---]   FOP.CMP.LT p0, R61, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R61, R61, R168
[NwwD---] FOP.MUL R52, R52, R175.l
[N--D---] FOP.MOV R52.l.bf16, R52.f32
[N--D---] FOP.MUL R53, R53, R175.l
[N--D---] FOP.MOV R53.l.bf16, R53.f32
[N--D---] FOP.MUL R60, R60, R175.l
[N--D---] FOP.MOV R60.l.bf16, R60.f32
[N--D---] FOP.MUL R61, R61, R175.l
[N-sD---] FOP.MOV R61.l.bf16, R61.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH92.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R52.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R53.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R60.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R61.l, SH[0:1], R158, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R158.s32, C45.s32, R158.s32
[NwwD---] FOP.FMA R54, R54, R185.l, R144
[N--D---] FOP.FMA R55, R55, R185.h, R145
[N--D---] FOP.FMA R62, R62, R186.l, R146
[N-sD---] FOP.FMA R63, R63, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R54, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R54, R54, R168
[N--D---]   FOP.CMP.LT p0, R55, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R55, R55, R168
[N--D---]   FOP.CMP.LT p0, R62, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R62, R62, R168
[N--D---]   FOP.CMP.LT p0, R63, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R63, R63, R168
[NwwD---] FOP.MUL R54, R54, R176.l
[N--D---] FOP.MOV R54.l.bf16, R54.f32
[N--D---] FOP.MUL R55, R55, R176.l
[N--D---] FOP.MOV R55.l.bf16, R55.f32
[N--D---] FOP.MUL R62, R62, R176.l
[N--D---] FOP.MOV R62.l.bf16, R62.f32
[N--D---] FOP.MUL R63, R63, R176.l
[N-sD---] FOP.MOV R63.l.bf16, R63.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH93.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R54.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R55.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R62.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R63.l, SH[0:1], R159, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R159.s32, C45.s32, R159.s32
[NwwD---] FOP.FMA R64, R64, R185.l, R144
[N--D---] FOP.FMA R65, R65, R185.h, R145
[N--D---] FOP.FMA R72, R72, R186.l, R146
[N-sD---] FOP.FMA R73, R73, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R64, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R64, R64, R168
[N--D---]   FOP.CMP.LT p0, R65, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R65, R65, R168
[N--D---]   FOP.CMP.LT p0, R72, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R72, R72, R168
[N--D---]   FOP.CMP.LT p0, R73, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R73, R73, R168
[NwwD---] FOP.MUL R64, R64, R177.l
[N--D---] FOP.MOV R64.l.bf16, R64.f32
[N--D---] FOP.MUL R65, R65, R177.l
[N--D---] FOP.MOV R65.l.bf16, R65.f32
[N--D---] FOP.MUL R72, R72, R177.l
[N--D---] FOP.MOV R72.l.bf16, R72.f32
[N--D---] FOP.MUL R73, R73, R177.l
[N-sD---] FOP.MOV R73.l.bf16, R73.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH95.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R64.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R65.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R72.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R73.l, SH[0:1], R160, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R160.s32, C45.s32, R160.s32
[NwwD---] FOP.FMA R66, R66, R185.l, R144
[N--D---] FOP.FMA R67, R67, R185.h, R145
[N--D---] FOP.FMA R74, R74, R186.l, R146
[N-sD---] FOP.FMA R75, R75, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R66, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R66, R66, R168
[N--D---]   FOP.CMP.LT p0, R67, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R67, R67, R168
[N--D---]   FOP.CMP.LT p0, R74, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R74, R74, R168
[N--D---]   FOP.CMP.LT p0, R75, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R75, R75, R168
[NwwD---] FOP.MUL R66, R66, R178.l
[N--D---] FOP.MOV R66.l.bf16, R66.f32
[N--D---] FOP.MUL R67, R67, R178.l
[N--D---] FOP.MOV R67.l.bf16, R67.f32
[N--D---] FOP.MUL R74, R74, R178.l
[N--D---] FOP.MOV R74.l.bf16, R74.f32
[N--D---] FOP.MUL R75, R75, R178.l
[N-sD---] FOP.MOV R75.l.bf16, R75.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH96.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R66.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R67.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R74.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R75.l, SH[0:1], R161, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R161.s32, C45.s32, R161.s32
[NwwD---] FOP.FMA R68, R68, R185.l, R144
[N--D---] FOP.FMA R69, R69, R185.h, R145
[N--D---] FOP.FMA R76, R76, R186.l, R146
[N-sD---] FOP.FMA R77, R77, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R68, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R68, R68, R168
[N--D---]   FOP.CMP.LT p0, R69, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R69, R69, R168
[N--D---]   FOP.CMP.LT p0, R76, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R76, R76, R168
[N--D---]   FOP.CMP.LT p0, R77, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R77, R77, R168
[NwwD---] FOP.MUL R68, R68, R179.l
[N--D---] FOP.MOV R68.l.bf16, R68.f32
[N--D---] FOP.MUL R69, R69, R179.l
[N--D---] FOP.MOV R69.l.bf16, R69.f32
[N--D---] FOP.MUL R76, R76, R179.l
[N--D---] FOP.MOV R76.l.bf16, R76.f32
[N--D---] FOP.MUL R77, R77, R179.l
[N-sD---] FOP.MOV R77.l.bf16, R77.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH97.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R68.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R69.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R76.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R77.l, SH[0:1], R162, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R162.s32, C45.s32, R162.s32
[NwwD---] FOP.FMA R70, R70, R185.l, R144
[N--D---] FOP.FMA R71, R71, R185.h, R145
[N--D---] FOP.FMA R78, R78, R186.l, R146
[N-sD---] FOP.FMA R79, R79, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R70, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R70, R70, R168
[N--D---]   FOP.CMP.LT p0, R71, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R71, R71, R168
[N--D---]   FOP.CMP.LT p0, R78, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R78, R78, R168
[N--D---]   FOP.CMP.LT p0, R79, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R79, R79, R168
[NwwD---] FOP.MUL R70, R70, R180.l
[N--D---] FOP.MOV R70.l.bf16, R70.f32
[N--D---] FOP.MUL R71, R71, R180.l
[N--D---] FOP.MOV R71.l.bf16, R71.f32
[N--D---] FOP.MUL R78, R78, R180.l
[N--D---] FOP.MOV R78.l.bf16, R78.f32
[N--D---] FOP.MUL R79, R79, R180.l
[N-sD---] FOP.MOV R79.l.bf16, R79.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH98.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R70.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R71.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R78.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R79.l, SH[0:1], R163, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R163.s32, C45.s32, R163.s32
[NwwD---] FOP.FMA R80, R80, R185.l, R144
[N--D---] FOP.FMA R81, R81, R185.h, R145
[N--D---] FOP.FMA R88, R88, R186.l, R146
[N-sD---] FOP.FMA R89, R89, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R80, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R80, R80, R168
[N--D---]   FOP.CMP.LT p0, R81, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R81, R81, R168
[N--D---]   FOP.CMP.LT p0, R88, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R88, R88, R168
[N--D---]   FOP.CMP.LT p0, R89, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R89, R89, R168
[NwwD---] FOP.MUL R80, R80, R181.l
[N--D---] FOP.MOV R80.l.bf16, R80.f32
[N--D---] FOP.MUL R81, R81, R181.l
[N--D---] FOP.MOV R81.l.bf16, R81.f32
[N--D---] FOP.MUL R88, R88, R181.l
[N--D---] FOP.MOV R88.l.bf16, R88.f32
[N--D---] FOP.MUL R89, R89, R181.l
[N-sD---] FOP.MOV R89.l.bf16, R89.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH99.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R80.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R81.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R88.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R89.l, SH[0:1], R164, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R164.s32, C45.s32, R164.s32
[NwwD---] FOP.FMA R82, R82, R185.l, R144
[N--D---] FOP.FMA R83, R83, R185.h, R145
[N--D---] FOP.FMA R90, R90, R186.l, R146
[N-sD---] FOP.FMA R91, R91, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R82, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R82, R82, R168
[N--D---]   FOP.CMP.LT p0, R83, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R83, R83, R168
[N--D---]   FOP.CMP.LT p0, R90, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R90, R90, R168
[N--D---]   FOP.CMP.LT p0, R91, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R91, R91, R168
[NwwD---] FOP.MUL R82, R82, R182.l
[N--D---] FOP.MOV R82.l.bf16, R82.f32
[N--D---] FOP.MUL R83, R83, R182.l
[N--D---] FOP.MOV R83.l.bf16, R83.f32
[N--D---] FOP.MUL R90, R90, R182.l
[N--D---] FOP.MOV R90.l.bf16, R90.f32
[N--D---] FOP.MUL R91, R91, R182.l
[N-sD---] FOP.MOV R91.l.bf16, R91.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH100.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R82.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R83.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R90.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R91.l, SH[0:1], R165, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R165.s32, C45.s32, R165.s32
[NwwD---] FOP.FMA R84, R84, R185.l, R144
[N--D---] FOP.FMA R85, R85, R185.h, R145
[N--D---] FOP.FMA R92, R92, R186.l, R146
[N-sD---] FOP.FMA R93, R93, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R84, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R84, R84, R168
[N--D---]   FOP.CMP.LT p0, R85, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R85, R85, R168
[N--D---]   FOP.CMP.LT p0, R92, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R92, R92, R168
[N--D---]   FOP.CMP.LT p0, R93, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R93, R93, R168
[NwwD---] FOP.MUL R84, R84, R183.l
[N--D---] FOP.MOV R84.l.bf16, R84.f32
[N--D---] FOP.MUL R85, R85, R183.l
[N--D---] FOP.MOV R85.l.bf16, R85.f32
[N--D---] FOP.MUL R92, R92, R183.l
[N--D---] FOP.MOV R92.l.bf16, R92.f32
[N--D---] FOP.MUL R93, R93, R183.l
[N-sD---] FOP.MOV R93.l.bf16, R93.f32
[NswD---]       INT.BYP.TG p0, R3.s32, SH101.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R84.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R85.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R92.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R93.l, SH[0:1], R166, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R166.s32, C45.s32, R166.s32
[NwwD---] FOP.FMA R86, R86, R185.l, R144
[N--D---] FOP.FMA R87, R87, R185.h, R145
[N--D---] FOP.FMA R94, R94, R186.l, R146
[N-sD---] FOP.FMA R95, R95, R186.h, R147
[N--D---]   FOP.CMP.LT p0, R86, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R86, R86, R168
[N--D---]   FOP.CMP.LT p0, R87, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R87, R87, R168
[N--D---]   FOP.CMP.LT p0, R94, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N--D---]   FOP.MUL R94, R94, R168
[N--D---]   FOP.CMP.LT p0, R95, C0
[N--D---]   FOP.MOVC R168, C1, SH65, p0
[N-sD---]   FOP.MUL R95, R95, R168
[NwwD---] FOP.MUL R86, R86, R184.l
[N--D---] FOP.MOV R86.l.bf16, R86.f32
[N--D---] FOP.MUL R87, R87, R184.l
[N--D---] FOP.MOV R87.l.bf16, R87.f32
[N--D---] FOP.MUL R94, R94, R184.l
[N--D---] FOP.MOV R94.l.bf16, R94.f32
[N--D---] FOP.MUL R95, R95, R184.l
[N-sD---] FOP.MOV R95.l.bf16, R95.f32
[NwsD---]       INT.BYP.TG p0, R3.s32, SH102.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R86.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R87.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NswD---]       INT.BYP.TG.AP p0, R4.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R94.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NwsD---]       INT.BYP.TG.AP p0, R4.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R95.l, SH[0:1], R167, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R167.s32, C45.s32, R167.s32
[NwwDwww]    CTRL.NOP
[NwwD---]    AP.BARRIER LM0
[NwwD---]    CTRL.END


.Lfunc_end1:
	.size	musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16, .Lfunc_end1-musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16




    .type	llvm.mtgpu.kernel.musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16.sm,@object
    .local	llvm.mtgpu.kernel.musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16.sm
    .comm	llvm.mtgpu.kernel.musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16.sm,20480,4

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
          .name:           output_buffer_size
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
          .align:          4
          .isVolatile:     false
          .layout:         0
          .name:           nr_batch
          .offset:         20
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
          .name:           icpg
          .offset:         21
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
          .name:           H
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
          .name:           W
          .offset:         23
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
          .name:           fh
          .offset:         24
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
          .name:           fw
          .offset:         25
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
          .name:           ocpg
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
          .name:           oh
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
          .name:           ow
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
          .name:           sh
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
          .name:           sw
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
          .name:           ph
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
          .name:           pw
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
          .name:           dh
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
          .name:           dw
          .offset:         34
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
          .name:           fast_ohw_ori
          .offset:         35
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
          .name:           fast_ohw_mul
          .offset:         36
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
          .name:           fast_ohw_sft
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
          .name:           fast_ow_ori
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
          .name:           fast_ow_mul
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
          .name:           fast_ow_sft
          .offset:         40
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
          .offset:         41
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
          .name:           oc
          .offset:         42
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
          .name:           ic_align_tilek
          .offset:         43
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
          .offset:         44
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
          .name:           fw_mul_dw
          .offset:         45
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
          .name:           rsc
          .offset:         46
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
          .name:           rs
          .offset:         47
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
          .name:           pa_stride
          .offset:         48
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
          .name:           pa_rewind
          .offset:         49
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
          .name:           pa_rewind_h
          .offset:         50
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
          .name:           pb_ocpg
          .offset:         51
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
          .name:           pb_step
          .offset:         52
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
          .name:           pb_stride
          .offset:         53
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
          .name:           pb_rewind
          .offset:         54
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
          .offset:         55
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
          .name:           icpg_sub_1
          .offset:         56
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
          .name:           icpg_sub_tilek
          .offset:         57
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
          .name:           batch_stride
          .offset:         58
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
          .name:           group
          .offset:         59
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
          .name:           with_bias
          .offset:         60
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
          .name:           rcp_scale_o
          .offset:         61
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
          .name:           act_mode
          .offset:         62
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
          .offset:         63
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
          .offset:         64
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
          .name:           leaky_relu_alpha
          .offset:         65
          .reg_bank:       3
          .size:           4
          .type:           8
        .input_mode:     0
        .kind:           18
      - .argBase:
          .address_space:  0
          .align:          16
          .isVolatile:     false
          .layout:         0
          .name:           a_scale_ptr
          .offset:         68
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
          .name:           a_scale_buffer_size
          .offset:         70
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
          .offset:         71
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
          .name:           b_scale_ptr
          .offset:         72
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
          .name:           b_scale_buffer_size
          .offset:         74
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
          .offset:         75
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
          .name:           prog_addr
          .offset:         76
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
          .name:           prog_size
          .offset:         78
          .reg_bank:       3
          .size:           4
          .type:           7
        .input_mode:     0
        .kind:           34
      - .argBase:
          .address_space:  0
          .align:          0
          .isVolatile:     false
          .layout:         0
          .name:           Global Spill Base
          .offset:         79
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
          .offset:         81
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
          .name:           gridDim_x
          .offset:         82
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
          .offset:         83
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
          .offset:         84
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
          .offset:         85
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
      .offset:         86
      .reg_bank:       3
      .size:           108
      .values:
        - 9
        - 10
        - 11
        - 13
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
        - 144
        - 160
        - 176
        - 192
        - 208
        - 224
        - 240
        - 16777215
    .indirect_call_flag: false
    .call_flag: true
    .internal_reg_count: 2
    .local_mem_used_in_cluster: 0
    .max_block_size: 1
    .name:           musa_asm_s8mma_nhwc_conv_KRSC_256_256x64_lin_ol_outbf16
    .private_memory_size: 0
    .shared_memory_size: 20608
    .shared_reg_count: 113
    .slot_reg_count: 4
    .temp_reg_count: 192
    .wave_mode:      0
mtgpu.version:
  - 2
  - 2
...

	.mtgpu_metadata.end

    