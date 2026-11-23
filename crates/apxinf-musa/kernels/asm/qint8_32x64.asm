
    .text
    
    .p2align    8
    .type reset_ic_idx,@function
    reset_ic_idx:
        
[Ns-D---]    FOP.MOV R14, C0
[N-sD---]    INT.ADD SL1.s32, C33.s32, SL1.s32
[N--D---]    INT.ADD R8.s32, SH34.s32, R8.s32
[N--D---]    INT.ADD R10.s32, SH49.s32, R10.s32
[Nw-D---]    INT.ADD R12.s32, SH54.s32, R12.s32
[NswD---]    INT.BYP.TL p0 SL1.s32, SH25.s32
[NwwD---]    CTRL.BR.ANYP0T BB0_s_end
[N--D---]    INT.ADD R8.s32, SH45.neg.s32, R8.s32
[N--D---]    INT.ADD R10.s32, SH50.s32, R10.s32
[Ns-D---]    INT.ADD R9.s32, SH33.s32, R9.s32
[NwsD---]    FOP.MOV SL1, C0
BB0_s_end:
[NswD---]    INT.BYP.TL p0, R9.u32, SH22.u32
[NwsD---]    INT.BYP.TL.AP p0, R8.u32, SH23.u32
[NswD---]    INT.BYP.MOVC.PRB R11.s32, R10.s32, SH6.s32, p0
[NwwD---]    CTRL.BASR SL0


    .Lfunc_end0:
    .size	reset_ic_idx, .Lfunc_end0-reset_ic_idx
        
	.text
	.protected	musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16
	.globl	musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16
	.p2align	8
	.type	musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16,@function
musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16:








    ; init stage
[N--D--o]   MOV.LD.LM.B32 R20, LM4
[N--Do--]   MOV.LD.LM.B32 R2, LM6
[NswD---]   FOP.MOV SL1, C0
[NwsD---]   FOP.MOV SL1, C0
[N--D--o]   MOV.LD.LM.B32 R21, LM5
[NswD---]   BIT.ROT.LSL SL3, C79, R20
[NwsD---]   BIT.ROT.LSL SL2, C80, R21
[NswD---]   BIT.LUT.AND R1, AR0, SH96
[NwsD---]   BIT.AND.LSL R22, C61, C50, AR0
[NswD---]   BIT.ROT.LSR R23, C45, AR0 
[NwsD---]   BIT.AND.LSL R25, C80, C33, AR0
[NswD---]   BIT.ROT.LSR R13, C33, AR0
[NwsD---]    FOP.MOV R14, C0
[NswD---]   INT.ADD R15.s32, SH21.s32, C0.s32
[NwsD---]   INT.ADD R27.s32, SL2.s32, R23.s32
[NswD---]   INT.BYP.TE p0, SH35.s32, C33.s32
[NwsD---] !p0 INT.MUL R[32:33].u64, SH36.u32, R27.u32
[NswD---] !p0 BIT.ROT.LSR R28, SH37, R33
[NwsD---]   INT.BYP.MOVC.PRB R28.s32, R27.s32, R28.s32, p0
[NswD---]   INT.MADD R29.s32, SH35.neg.s32, R27.s32, R28.s32
[NwsD---]   INT.BYP.TE p0, SH38.s32, C33.s32
[NswD---] !p0 INT.MUL R[32:33].u64, SH39.u32, R29.u32
[NwsD---] !p0 BIT.ROT.LSR R30, SH40, R33
[NswD---]   INT.BYP.MOVC.PRB R30.s32, R29.s32, R30.s32, p0
[NwsD---]   INT.MADD R31.s32, SH38.neg.s32, R29.s32, R30.s32
[NswD---]   INT.MADD R9.s32, SH29.s32, SH31.neg.s32, R30.s32
[NwsD---]   INT.MADD R8.s32, SH30.s32, SH32.neg.s32, R31.s32
[NswD---]   INT.BYP.TL p0, R9.u32, SH22.u32
[NwsD---]   INT.BYP.TL.AP p0, R8.u32, SH23.u32
[NswD---]     INT.MADD R10.s32, SH22.s32, R9.s32, R28.s32
[NwsD---]     INT.MADD R10.s32, SH23.s32, R8.s32, R10.s32
[NswD---]     INT.MADD R10.s32, SH59.s32, R2.s32, R10.s32
[NwsD---]     INT.MADD R10.s32, SH21.s32, R22.s32, R10.s32
[NswD---]     BIT.ROT.LSL R10, C0, R10
[NwsD---]   INT.BYP.MOVC.PRB R11.s32, R10.s32, SH6.s32, p0
[NswD---]       INT.ADD R12.s32, SL3.s32, R13.s32
[NwsD---]       INT.MADD R12.s32, SH51.s32, R12.s32, R2.s32
[NswD---]       INT.MADD R12.s32, SH46.s32, R25.s32, R12.s32
[NwsD---]       BIT.ROT.LSL R12, C0, R12
[NswD---]       BIT.ROT.LSR R3, C79, R1
[NwsD---]       BIT.ROT.LSL R3, C61, R3
[NswD---]       BIT.AND.LSL R19, C45, C50, R1
[NwsD---]       INT.ADD R3.s32, R19.s32, R3.s32
[NswD---]       INT.ADD R19.s32, C33.s32, R3.s32
[NwsD---]       INT.ADD R22.s32, C45.s32, R3.s32
[NswD---]       INT.ADD R23.s32, C50.s32, R3.s32
[NwsD---]       BIT.LUT.AND R30, SH91, SH21
[NswD---]       INT.BYP.TE p0, R30.s32, C0.s32
[NwsD---]       INT.BYP.MOVC.PRB R30.s32, SH92.s32, R30.s32, p0
[NswD---]       INT.BYP.TL p0, R3.s32, R30.s32
[NwsD---]       INT.BYP.MOVC.PRB R4.s32, C34.s32, C0.s32, p0
[NswD---]       INT.BYP.TL p0, R19.s32, R30.s32
[NwsD---]       INT.BYP.MOVC.PRB R4.s32, C52.s32, R4.s32, p0
[NswD---]       INT.BYP.TL p0, R22.s32, R30.s32
[NwsD---]       INT.BYP.MOVC.PRB R4.s32, SH105.s32, R4.s32, p0
[NswD---]       INT.BYP.TL p0, R23.s32, R30.s32
[NwsD---]       INT.BYP.MOVC.PRB R4.s32, C36.s32, R4.s32, p0
[NswD---]       INT.ADD R30.s32, R30.s32, SH89.neg.s32
[NwsD---]       INT.BYP.TL p0, R3.s32, R30.s32
[NswD---]       INT.BYP.MOVC.PRB R5.s32, C34.s32, C0.s32, p0
[NwsD---]       INT.BYP.TL p0, R19.s32, R30.s32
[NswD---]       INT.BYP.MOVC.PRB R5.s32, C52.s32, R5.s32, p0
[NwsD---]       INT.BYP.TL p0, R22.s32, R30.s32
[NswD---]       INT.BYP.MOVC.PRB R5.s32, SH105.s32, R5.s32, p0
[NwsD---]       INT.BYP.TL p0, R23.s32, R30.s32
[NswD---]       INT.BYP.MOVC.PRB R5.s32, C36.s32, R5.s32, p0
[NwsD---]       INT.BYP.TL p0, R14.s32, SH57.s32
[NswD---]       INT.BYP.MOVC.PRB R3.s32, C33.s32, C0.s32, p0
[NwwDs--]       DMA.LD.B128 R[194:197], SH[4:5], R11, _, _, _, _,  SH[4:6]
[N--Ds--]       DMA.LD.B256 R[198:205], SH[8:9], R12, _, _, _, _,  SH[8:10]
[NwsD---]       BIT.AND.LSR R25, C33, C50, R1
[NswD---]       BIT.ROT.LSL R25, C33, R25
[NwsD---]       BIT.ROT.LSR R19, C79, R1
[NswD---]       INT.ADD R25.s32, R25.s32, R19.s32
[NwsD---]       BIT.ROT.LSL R25, SH86, R25
[NswD---]       BIT.AND.LSL R26, C46, C33, R1
[NwsD---]       BIT.AND.LSR R28, C45, SH91, R1
[NswD---]       BIT.ROT.LSL R28, C61, R28
[NwsD---]       INT.ADD R17.s32, R25.s32, R26.s32
[NswD---]       INT.ADD R17.s32, R17.s32, R28.s32
[NwsD---]       BIT.ROT.LSR R17, C45, R17
[NswD---]       BIT.ROT.LSR R19, C45, R13
[NwsD---]       BIT.AND.LSL R13, C61, C50, R13
[NswD---]       INT.ADD R13.s32, R19.s32, R13.s32
[NwsD---]       BIT.AND.LSL R23, SH88, C33, AR0
[NswD---]       BIT.ROT.LSR R25, C61, R13
[NwsD---]       BIT.ROT.LSL R25, SH86, R25
[NswD---]       INT.ADD R25.s32, R25.s32, R23.s32
[NwsD---]       BIT.AND.LSR R28, C0, C111, R13
[NswD---]       BIT.ROT.LSL R28, C61, R28
[NwsD---]       INT.ADD R18.s32, R25.s32, R28.s32
[NswD---]       BIT.ROT.LSR R18, C45, R18
[NwsD---]       BIT.ROT.LSL R6, C0, R1
[NswD---]       BIT.ROT.LSL R7, C0, R1
[NwsD---]       INT.ADD R16.s32, SH44.s32, SH92.neg.s32
[N--D---] FOP.MOV R32, C0
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
[N--Dw--] FOP.MOV R45, C0
[N--D---] FOP.MOV R46, C0
[N-sD---] FOP.MOV R47, C0

[N-wD-s-]       LDMA.WR.B128.BST1 R[194:197], R17, X4, 128 , BURST_STRIDE=16
[N--D-s-]       LDMA.WR.B128.BST2 R[198:205], R18, X4, 2176 , BURST_STRIDE=256
[N-wD---]       INT.ADD R11.s32, SH48.s32, R11.s32
[N-sD-w-]       INT.ADD R12.s32, SH53.s32, R12.s32
[NswD---]       INT.ADD R14.s32, SH92.s32, R14.s32
[NwsD---]       INT.BYP.TL p0, R14.s32, R15.s32
[NwwD---]       CTRL.BR.ANYP0T BB0_no_reset_ic_pre
[NwwD---]       CTRL.BR SL0, reset_ic_idx@musarel32@dword
BB0_no_reset_ic_pre:
[N--D-s-]       LDMA.RD.B32.BST4  R[160:163], R6, X4, 128, BURST_STRIDE=512
[N--D-s-]       LDMA.RD.B32.BST4  R[164:167], R6, X4, 2176, BURST_STRIDE=512
[N--D-o-]       LDMA.RD.B32.BST4  R[177:180], R7, X4, 2176, BURST_STRIDE=512
[NwsD---]       INT.BYP.TE p0, R3.s32, C0.s32
[N-wD---]   P0    BIT.LUT.AND R177, R177, R4
[N--D---]   P0    BIT.LUT.AND R178, R178, R4
[N--D---]   P0    BIT.LUT.AND R179, R179, R4
[N-sD---]   P0    BIT.LUT.AND R180, R180, R4
[NswD--w]       INT.BYP.TG p0, R16.s32, C0.s32
[Nw-D---]       CTRL.BR.ALLP0F LAST_LOOP


MAIN_LOOP:

[N--Ds--]       DMA.LD.B128 R[194:197], SH[4:5], R11, _, _, _, _,  SH[4:6]
[N--D---] MMA.323232 R32[4:2][2:1].s32, R160[2:1][1:1].s8.row, R177[1:1][2:1].s8.col, R32[4:2][2:1].s32
[N--Ds--]       DMA.LD.B256 R[198:205], SH[8:9], R12, _, _, _, _,  SH[8:10]
[Ns-Dw--] MMA.323232 R40[4:2][2:1].s32, R160[2:1][1:1].s8.row, R179[1:1][2:1].s8.col, R40[4:2][2:1].s32
[N-wD--s]       LDMA.RD.B32.BST4  R[168:171], R6, X4, 1152, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[172:175], R6, X4, 3200, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[185:188], R7, X4, 4224, BURST_STRIDE=512
[NwsD--w]       INT.BYP.TE p0, R3.s32, C0.s32
[N-wD---]   P0    BIT.LUT.AND R185, R185, R5
[N--D---]   P0    BIT.LUT.AND R186, R186, R5
[N--D---]   P0    BIT.LUT.AND R187, R187, R5
[Ns-D---]   P0    BIT.LUT.AND R188, R188, R5
[Nw-D---]       CTRL.NOP
[N--D---] MMA.323232 R32[4:2][2:1].s32, R168[2:1][1:1].s8.row, R185[1:1][2:1].s8.col, R32[4:2][2:1].s32
[NwwD--s]       LDMA.WR.B128.BST1 R[194:197], R17, X4, 128 , BURST_STRIDE=16
[N-sD---] MMA.323232 R40[4:2][2:1].s32, R168[2:1][1:1].s8.row, R187[1:1][2:1].s8.col, R40[4:2][2:1].s32
[N--D--s]       LDMA.WR.B128.BST2 R[198:205], R18, X4, 2176 , BURST_STRIDE=256
[N-wD---]       INT.ADD R11.s32, SH48.s32, R11.s32
[N-sD---]       INT.ADD R12.s32, SH53.s32, R12.s32
[N-wD--s]       LDMA.RD.B32.BST4  R[160:163], R6, X4, 128, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[164:167], R6, X4, 2176, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[177:180], R7, X4, 2176, BURST_STRIDE=512
[NwsD--w]       INT.BYP.TL p0, R14.s32, SH57.s32
[NswD---]       INT.BYP.MOVC.PRB R3.s32, C33.s32, C0.s32, p0
[NwsD---]       INT.BYP.TE p0, R3.s32, C0.s32
[N-wD---]   P0    BIT.LUT.AND R177, R177, R4
[N--D---]   P0    BIT.LUT.AND R178, R178, R4
[N--D---]   P0    BIT.LUT.AND R179, R179, R4
[Ns-D---]   P0    BIT.LUT.AND R180, R180, R4
[Ns-D---]    INT.ADD R14.s32, SH92.s32, R14.s32
[NwsD---]    INT.BYP.TL p0, R14.s32, R15.s32
[NwwD---]    CTRL.BR.ANYP0T BB0_no_reset_ic
[NwwD---]    CTRL.BR SL0, reset_ic_idx@musarel32@dword
BB0_no_reset_ic:
[Ns-D---]    INT.ADD R16.s32, R16.s32, SH92.neg.s32
[NwsD---]    INT.BYP.TG p0, R16.s32, C0.s32
[N-wD---]    CTRL.BR.ANYP0T MAIN_LOOP


LAST_LOOP:
[N-wD--s]       LDMA.RD.B32.BST4  R[168:171], R6, X4, 1152, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[172:175], R6, X4, 3200, BURST_STRIDE=512
[N--D--s]       LDMA.RD.B32.BST4  R[185:188], R7, X4, 4224, BURST_STRIDE=512
[N-wD---] MMA.323232 R32[4:2][2:1].s32, R160[2:1][1:1].s8.row, R177[1:1][2:1].s8.col, R32[4:2][2:1].s32
[NswD---]       BIT.ROT.LSR R18, C79, R1
[NwsD---]       BIT.ROT.LSL R18, C45, R18
[NswD---]       BIT.AND.LSR R17, C50, C78, R1
[NwsD---]       INT.ADD R18.s32, R17.s32, R18.s32
[NswD---]       INT.ADD R18.s32, SL2.s32, R18.s32
[Ns-D--w] MMA.323232 R40[4:2][2:1].s32, R160[2:1][1:1].s8.row, R179[1:1][2:1].s8.col, R40[4:2][2:1].s32
[NwsD---]       BIT.AND.LSR R15, C80, SH91, R1
[NswD---]       BIT.ROT.LSL R15, C80, R15
[NwsD---]       BIT.AND.LSL R17, C45, C59, R1
[NswD---]       INT.ADD R15.s32, R17.s32, R15.s32
[NwsD---]       INT.ADD R15.s32, SL3.s32, R15.s32
[NswD---]       INT.ADD R9.s32, SH26.s32, R15.neg.s32
[NwsD---]       INT.MADD R15.s32, SH26.s32, R15.s32, R2.s32
[NswD---]       INT.MADD R19.s32, SH42.s32, R15.s32, R18.s32
[NwsD---]       BIT.ROT.LSL R19, C33, R19
[NswD---]       FOP.MOV R20, SH42
[NwsD---]       INT.ADD R8.s32, SH41.s32, R18.neg.s32
[NswD---]       INT.MUL R16.s32, SH92.s32,  R20.s32
[NwwD---]       INT.MADD R216.s32, C0.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R217.s32, C39.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R218.s32, SH89.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R219.s32, SH90.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R220.s32, SH92.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R221.s32, SH93.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R222.s32, SH94.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R223.s32, SH95.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R224.s32, SH97.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R225.s32, SH98.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R226.s32, SH99.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R227.s32, SH100.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R228.s32, SH101.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R229.s32, SH102.s32, R19.s32, R20.s32
[N--D---]       INT.MADD R230.s32, SH103.s32, R19.s32, R20.s32
[Ns-D---]       INT.MADD R231.s32, SH104.s32, R19.s32, R20.s32
[N--D---]       FOP.MOV R208, C0
[N--D---]       FOP.MOV R209, C0
[N--D---]       FOP.MOV R210, C0
[N--D---]       FOP.MOV R211, C0
[N--D---]       FOP.MOV R212, C0
[N--D---]       FOP.MOV R213, C0
[N--D---]       FOP.MOV R214, C0
[Ns-D---]       FOP.MOV R215, C0
[Ns-D---]        INT.BYP.TE p0, SH60.s32, C33.s32
[NwsD---]  P0  BIT.ROT.LSL R208, C45, R15
[N-wD-s-]  P0      DMA.LD.B256 R[208:215], SH[12:13], R208, _, _, _, _,  SH[12:14]
[NwwD---]       BIT.LUT.AND R185, R185, R5
[N--D---]       BIT.LUT.AND R186, R186, R5
[N--D---]       BIT.LUT.AND R187, R187, R5
[N-sD---]       BIT.LUT.AND R188, R188, R5
[N-wD---] MMA.323232 R32[4:2][2:1].s32, R168[2:1][1:1].s8.row, R185[1:1][2:1].s8.col, R32[4:2][2:1].s32
[N-sD-w-] MMA.323232 R40[4:2][2:1].s32, R168[2:1][1:1].s8.row, R187[1:1][2:1].s8.col, R40[4:2][2:1].s32
[N-wD---]       FOP.MUL R208, R208, SH61
[N--D---]       FOP.MUL R209, R209, SH61
[N--D---]       FOP.MUL R210, R210, SH61
[N--D---]       FOP.MUL R211, R211, SH61
[N--D---]       FOP.MUL R212, R212, SH61
[N--D---]       FOP.MUL R213, R213, SH61
[N--D---]       FOP.MUL R214, R214, SH61
[N-sD---]       FOP.MUL R215, R215, SH61
; core loop stage end







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
[N-sD---]     MOV.S32_TO_F32_RTE R47, R47
[NswD---]       BIT.ROT.LSL R16, C33, R15
[NwsD---]       BIT.ROT.LSL R10, C33, R18
[NwwD-s-]       DMA.LD.B64 R[192:193], SH[72:73], R16, _, _, _, _,  SH[72:74]
[NswD---]       INT.ADD R11.s32, C0.s32, R10.s32
[NwsD---]       INT.ADD R12.s32, C39.s32, R10.s32
[NswD---]       INT.ADD R13.s32, SH89.s32, R10.s32
[NwsD---]       INT.ADD R14.s32, SH90.s32, R10.s32
[NwwD---]       CTRL.NOP
[N--D-s-]       DMA.LD.B16 R160.l, SH[68:69], R11, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R161.l, SH[68:69], R12, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R162.l, SH[68:69], R13, _, _, _, _,  SH[68:70]
[N--D-s-]       DMA.LD.B16 R163.l, SH[68:69], R14, _, _, _, _,  SH[68:70]
[NwwDwww]       CTRL.NOP
[NswD---]     INT.BYP.TE p0, SH62.s32, C0.s32
[Nw-D---]     CTRL.BR.ANYP0T IDENTITY_START
[NswD---]     INT.BYP.TE p0, SH62.s32, C45.s32
[Nw-D---]     CTRL.BR.ANYP0T RELU_START
[NswD---]     INT.BYP.TE p0, SH62.s32, SH87.s32
[Nw-D---]     CTRL.BR.ANYP0T LEAKYRELU_START
IDENTITY_START:
[NswD---]       INT.BYP.TLE p0, R8.s32, C48.s32
[NwsD---]       INT.BYP.TLE.OP p0, R9.s32, C59.s32
[NwwD---]        CTRL.BR.ANYP0T STORE_MASK_IDENTITY
[N-wD---] FOP.FMA R32, R32, R192.l, R208
[N--D---] FOP.FMA R33, R33, R192.h, R209
[N--D---] FOP.FMA R40, R40, R193.l, R210
[N-sD---] FOP.FMA R41, R41, R193.h, R211
[N--D---] FOP.MUL R32, R160.l, R32
[N--D---] FOP.MOV R176.l.bf16, R32.f32
[N--D---] FOP.MUL R40, R160.l, R40
[N--D---] FOP.MOV R177.l.bf16, R40.f32
[N--D---] FOP.MUL R33, R160.l, R33
[N--D---] FOP.MOV R176.h.bf16, R33.f32
[N--D---] FOP.MUL R41, R160.l, R41
[N-sD---] FOP.MOV R177.h.bf16, R41.f32
[NwwD---]       DMA.ST.B64 R[176:177], SH[0:1], R216, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R34, R34, R192.l, R208
[N--D---] FOP.FMA R35, R35, R192.h, R209
[N--D---] FOP.FMA R42, R42, R193.l, R210
[N-sD---] FOP.FMA R43, R43, R193.h, R211
[N--D---] FOP.MUL R34, R161.l, R34
[N--D---] FOP.MOV R178.l.bf16, R34.f32
[N--D---] FOP.MUL R42, R161.l, R42
[N--D---] FOP.MOV R179.l.bf16, R42.f32
[N--D---] FOP.MUL R35, R161.l, R35
[N--D---] FOP.MOV R178.h.bf16, R35.f32
[N--D---] FOP.MUL R43, R161.l, R43
[N-sD---] FOP.MOV R179.h.bf16, R43.f32
[NwwD---]       DMA.ST.B64 R[178:179], SH[0:1], R217, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R36, R36, R192.l, R208
[N--D---] FOP.FMA R37, R37, R192.h, R209
[N--D---] FOP.FMA R44, R44, R193.l, R210
[N-sD---] FOP.FMA R45, R45, R193.h, R211
[N--D---] FOP.MUL R36, R162.l, R36
[N--D---] FOP.MOV R180.l.bf16, R36.f32
[N--D---] FOP.MUL R44, R162.l, R44
[N--D---] FOP.MOV R181.l.bf16, R44.f32
[N--D---] FOP.MUL R37, R162.l, R37
[N--D---] FOP.MOV R180.h.bf16, R37.f32
[N--D---] FOP.MUL R45, R162.l, R45
[N-sD---] FOP.MOV R181.h.bf16, R45.f32
[NwwD---]       DMA.ST.B64 R[180:181], SH[0:1], R218, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R38, R38, R192.l, R208
[N--D---] FOP.FMA R39, R39, R192.h, R209
[N--D---] FOP.FMA R46, R46, R193.l, R210
[N-sD---] FOP.FMA R47, R47, R193.h, R211
[N--D---] FOP.MUL R38, R163.l, R38
[N--D---] FOP.MOV R182.l.bf16, R38.f32
[N--D---] FOP.MUL R46, R163.l, R46
[N--D---] FOP.MOV R183.l.bf16, R46.f32
[N--D---] FOP.MUL R39, R163.l, R39
[N--D---] FOP.MOV R182.h.bf16, R39.f32
[N--D---] FOP.MUL R47, R163.l, R47
[N-sD---] FOP.MOV R183.h.bf16, R47.f32
[NswD---]       DMA.ST.B64 R[182:183], SH[0:1], R219, _, _, _, _,  SH[0:2]
[Nw-Dwww]    CTRL.NOP
[N-wD---]    CTRL.END
STORE_MASK_IDENTITY:
[NwwD---] FOP.FMA R32, R32, R192.l, R208
[N--D---] FOP.FMA R33, R33, R192.h, R209
[N--D---] FOP.FMA R40, R40, R193.l, R210
[N-sD---] FOP.FMA R41, R41, R193.h, R211
[NwwD---] FOP.MUL R32, R32, R160.l
[N--D---] FOP.MOV R32.l.bf16, R32.f32
[N--D---] FOP.MUL R33, R33, R160.l
[N--D---] FOP.MOV R33.l.bf16, R33.f32
[N--D---] FOP.MUL R40, R40, R160.l
[N--D---] FOP.MOV R40.l.bf16, R40.f32
[N--D---] FOP.MUL R41, R41, R160.l
[N-sD---] FOP.MOV R41.l.bf16, R41.f32
[NswD---]       INT.BYP.TG p0, R8.s32, C0.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R32.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R33.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R40.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R41.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NwwD---] FOP.FMA R34, R34, R192.l, R208
[N--D---] FOP.FMA R35, R35, R192.h, R209
[N--D---] FOP.FMA R42, R42, R193.l, R210
[N-sD---] FOP.FMA R43, R43, R193.h, R211
[NwwD---] FOP.MUL R34, R34, R161.l
[N--D---] FOP.MOV R34.l.bf16, R34.f32
[N--D---] FOP.MUL R35, R35, R161.l
[N--D---] FOP.MOV R35.l.bf16, R35.f32
[N--D---] FOP.MUL R42, R42, R161.l
[N--D---] FOP.MOV R42.l.bf16, R42.f32
[N--D---] FOP.MUL R43, R43, R161.l
[N-sD---] FOP.MOV R43.l.bf16, R43.f32
[NwsD---]       INT.BYP.TG p0, R8.s32, C46.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R34.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R35.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R42.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R43.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NwwD---] FOP.FMA R36, R36, R192.l, R208
[N--D---] FOP.FMA R37, R37, R192.h, R209
[N--D---] FOP.FMA R44, R44, R193.l, R210
[N-sD---] FOP.FMA R45, R45, R193.h, R211
[NwwD---] FOP.MUL R36, R36, R162.l
[N--D---] FOP.MOV R36.l.bf16, R36.f32
[N--D---] FOP.MUL R37, R37, R162.l
[N--D---] FOP.MOV R37.l.bf16, R37.f32
[N--D---] FOP.MUL R44, R44, R162.l
[N--D---] FOP.MOV R44.l.bf16, R44.f32
[N--D---] FOP.MUL R45, R45, R162.l
[N-sD---] FOP.MOV R45.l.bf16, R45.f32
[NswD---]       INT.BYP.TG p0, R8.s32, C39.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R36.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R37.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R44.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R45.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NwwD---] FOP.FMA R38, R38, R192.l, R208
[N--D---] FOP.FMA R39, R39, R192.h, R209
[N--D---] FOP.FMA R46, R46, R193.l, R210
[N-sD---] FOP.FMA R47, R47, R193.h, R211
[NwwD---] FOP.MUL R38, R38, R163.l
[N--D---] FOP.MOV R38.l.bf16, R38.f32
[N--D---] FOP.MUL R39, R39, R163.l
[N--D---] FOP.MOV R39.l.bf16, R39.f32
[N--D---] FOP.MUL R46, R46, R163.l
[N--D---] FOP.MOV R46.l.bf16, R46.f32
[N--D---] FOP.MUL R47, R47, R163.l
[N-sD---] FOP.MOV R47.l.bf16, R47.f32
[NwsD---]       INT.BYP.TG p0, R8.s32, C48.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R38.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R219.s32, C45.s32, R219.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R39.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R219.s32, C45.s32, R219.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R46.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R219.s32, C45.s32, R219.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R47.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R219.s32, C45.s32, R219.s32
[Nw-Dwww]    CTRL.NOP
[N-wD---]    CTRL.END
RELU_START:
[NswD---]       INT.BYP.TLE p0, R8.s32, C48.s32
[NwsD---]       INT.BYP.TLE.OP p0, R9.s32, C59.s32
[NwwD---]        CTRL.BR.ANYP0T STORE_MASK_RELU
[N-wD---] FOP.FMA R32, R32, R192.l, R208
[N--D---] FOP.FMA R33, R33, R192.h, R209
[N--D---] FOP.FMA R40, R40, R193.l, R210
[N-sD---] FOP.FMA R41, R41, R193.h, R211
[N--D---]    FOP.CMP_MOVC.MAX.TST R32, R32, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R33, R33, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R40, R40, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R41, R41, C0
[N--D---] FOP.MUL R32, R160.l, R32
[N--D---] FOP.MOV R176.l.bf16, R32.f32
[N--D---] FOP.MUL R40, R160.l, R40
[N--D---] FOP.MOV R177.l.bf16, R40.f32
[N--D---] FOP.MUL R33, R160.l, R33
[N--D---] FOP.MOV R176.h.bf16, R33.f32
[N--D---] FOP.MUL R41, R160.l, R41
[N-sD---] FOP.MOV R177.h.bf16, R41.f32
[NwwD---]       DMA.ST.B64 R[176:177], SH[0:1], R216, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R34, R34, R192.l, R208
[N--D---] FOP.FMA R35, R35, R192.h, R209
[N--D---] FOP.FMA R42, R42, R193.l, R210
[N-sD---] FOP.FMA R43, R43, R193.h, R211
[N--D---]    FOP.CMP_MOVC.MAX.TST R34, R34, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R35, R35, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R42, R42, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R43, R43, C0
[N--D---] FOP.MUL R34, R161.l, R34
[N--D---] FOP.MOV R178.l.bf16, R34.f32
[N--D---] FOP.MUL R42, R161.l, R42
[N--D---] FOP.MOV R179.l.bf16, R42.f32
[N--D---] FOP.MUL R35, R161.l, R35
[N--D---] FOP.MOV R178.h.bf16, R35.f32
[N--D---] FOP.MUL R43, R161.l, R43
[N-sD---] FOP.MOV R179.h.bf16, R43.f32
[NwwD---]       DMA.ST.B64 R[178:179], SH[0:1], R217, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R36, R36, R192.l, R208
[N--D---] FOP.FMA R37, R37, R192.h, R209
[N--D---] FOP.FMA R44, R44, R193.l, R210
[N-sD---] FOP.FMA R45, R45, R193.h, R211
[N--D---]    FOP.CMP_MOVC.MAX.TST R36, R36, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R37, R37, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R44, R44, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R45, R45, C0
[N--D---] FOP.MUL R36, R162.l, R36
[N--D---] FOP.MOV R180.l.bf16, R36.f32
[N--D---] FOP.MUL R44, R162.l, R44
[N--D---] FOP.MOV R181.l.bf16, R44.f32
[N--D---] FOP.MUL R37, R162.l, R37
[N--D---] FOP.MOV R180.h.bf16, R37.f32
[N--D---] FOP.MUL R45, R162.l, R45
[N-sD---] FOP.MOV R181.h.bf16, R45.f32
[NwwD---]       DMA.ST.B64 R[180:181], SH[0:1], R218, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R38, R38, R192.l, R208
[N--D---] FOP.FMA R39, R39, R192.h, R209
[N--D---] FOP.FMA R46, R46, R193.l, R210
[N-sD---] FOP.FMA R47, R47, R193.h, R211
[N--D---]    FOP.CMP_MOVC.MAX.TST R38, R38, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R39, R39, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R46, R46, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R47, R47, C0
[N--D---] FOP.MUL R38, R163.l, R38
[N--D---] FOP.MOV R182.l.bf16, R38.f32
[N--D---] FOP.MUL R46, R163.l, R46
[N--D---] FOP.MOV R183.l.bf16, R46.f32
[N--D---] FOP.MUL R39, R163.l, R39
[N--D---] FOP.MOV R182.h.bf16, R39.f32
[N--D---] FOP.MUL R47, R163.l, R47
[N-sD---] FOP.MOV R183.h.bf16, R47.f32
[NswD---]       DMA.ST.B64 R[182:183], SH[0:1], R219, _, _, _, _,  SH[0:2]
[Nw-Dwww]    CTRL.NOP
[N-wD---]    CTRL.END
STORE_MASK_RELU:
[NwwD---] FOP.FMA R32, R32, R192.l, R208
[N--D---] FOP.FMA R33, R33, R192.h, R209
[N--D---] FOP.FMA R40, R40, R193.l, R210
[N-sD---] FOP.FMA R41, R41, R193.h, R211
[N--D---]    FOP.CMP_MOVC.MAX.TST R32, R32, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R33, R33, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R40, R40, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R41, R41, C0
[NwwD---] FOP.MUL R32, R32, R160.l
[N--D---] FOP.MOV R32.l.bf16, R32.f32
[N--D---] FOP.MUL R33, R33, R160.l
[N--D---] FOP.MOV R33.l.bf16, R33.f32
[N--D---] FOP.MUL R40, R40, R160.l
[N--D---] FOP.MOV R40.l.bf16, R40.f32
[N--D---] FOP.MUL R41, R41, R160.l
[N-sD---] FOP.MOV R41.l.bf16, R41.f32
[NswD---]       INT.BYP.TG p0, R8.s32, C0.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R32.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R33.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R40.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R41.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NwwD---] FOP.FMA R34, R34, R192.l, R208
[N--D---] FOP.FMA R35, R35, R192.h, R209
[N--D---] FOP.FMA R42, R42, R193.l, R210
[N-sD---] FOP.FMA R43, R43, R193.h, R211
[N--D---]    FOP.CMP_MOVC.MAX.TST R34, R34, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R35, R35, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R42, R42, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R43, R43, C0
[NwwD---] FOP.MUL R34, R34, R161.l
[N--D---] FOP.MOV R34.l.bf16, R34.f32
[N--D---] FOP.MUL R35, R35, R161.l
[N--D---] FOP.MOV R35.l.bf16, R35.f32
[N--D---] FOP.MUL R42, R42, R161.l
[N--D---] FOP.MOV R42.l.bf16, R42.f32
[N--D---] FOP.MUL R43, R43, R161.l
[N-sD---] FOP.MOV R43.l.bf16, R43.f32
[NwsD---]       INT.BYP.TG p0, R8.s32, C46.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R34.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R35.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R42.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R43.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NwwD---] FOP.FMA R36, R36, R192.l, R208
[N--D---] FOP.FMA R37, R37, R192.h, R209
[N--D---] FOP.FMA R44, R44, R193.l, R210
[N-sD---] FOP.FMA R45, R45, R193.h, R211
[N--D---]    FOP.CMP_MOVC.MAX.TST R36, R36, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R37, R37, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R44, R44, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R45, R45, C0
[NwwD---] FOP.MUL R36, R36, R162.l
[N--D---] FOP.MOV R36.l.bf16, R36.f32
[N--D---] FOP.MUL R37, R37, R162.l
[N--D---] FOP.MOV R37.l.bf16, R37.f32
[N--D---] FOP.MUL R44, R44, R162.l
[N--D---] FOP.MOV R44.l.bf16, R44.f32
[N--D---] FOP.MUL R45, R45, R162.l
[N-sD---] FOP.MOV R45.l.bf16, R45.f32
[NswD---]       INT.BYP.TG p0, R8.s32, C39.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R36.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R37.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R44.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R45.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NwwD---] FOP.FMA R38, R38, R192.l, R208
[N--D---] FOP.FMA R39, R39, R192.h, R209
[N--D---] FOP.FMA R46, R46, R193.l, R210
[N-sD---] FOP.FMA R47, R47, R193.h, R211
[N--D---]    FOP.CMP_MOVC.MAX.TST R38, R38, C0
[N--D---]   FOP.CMP_MOVC.MAX.TST R39, R39, C0
[N--D---]    FOP.CMP_MOVC.MAX.TST R46, R46, C0
[N-sD---]   FOP.CMP_MOVC.MAX.TST R47, R47, C0
[NwwD---] FOP.MUL R38, R38, R163.l
[N--D---] FOP.MOV R38.l.bf16, R38.f32
[N--D---] FOP.MUL R39, R39, R163.l
[N--D---] FOP.MOV R39.l.bf16, R39.f32
[N--D---] FOP.MUL R46, R46, R163.l
[N--D---] FOP.MOV R46.l.bf16, R46.f32
[N--D---] FOP.MUL R47, R47, R163.l
[N-sD---] FOP.MOV R47.l.bf16, R47.f32
[NwsD---]       INT.BYP.TG p0, R8.s32, C48.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R38.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R219.s32, C45.s32, R219.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R39.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R219.s32, C45.s32, R219.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R46.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R219.s32, C45.s32, R219.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R47.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R219.s32, C45.s32, R219.s32
[Nw-Dwww]    CTRL.NOP
[N-wD---]    CTRL.END
LEAKYRELU_START:
[NswD---]       INT.BYP.TLE p0, R8.s32, C48.s32
[NwsD---]       INT.BYP.TLE.OP p0, R9.s32, C59.s32
[NwwD---]        CTRL.BR.ANYP0T STORE_MASK_LEAKYRELU
[N-wD---] FOP.FMA R32, R32, R192.l, R208
[N--D---] FOP.FMA R33, R33, R192.h, R209
[N--D---] FOP.FMA R40, R40, R193.l, R210
[N-sD---] FOP.FMA R41, R41, R193.h, R211
[N--D---]   FOP.CMP.LT p0, R32, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R32, R32, R232
[N--D---]   FOP.CMP.LT p0, R33, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R33, R33, R232
[N--D---]   FOP.CMP.LT p0, R40, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R40, R40, R232
[N--D---]   FOP.CMP.LT p0, R41, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N-sD---]   FOP.MUL R41, R41, R232
[N--D---] FOP.MUL R32, R160.l, R32
[N--D---] FOP.MOV R176.l.bf16, R32.f32
[N--D---] FOP.MUL R40, R160.l, R40
[N--D---] FOP.MOV R177.l.bf16, R40.f32
[N--D---] FOP.MUL R33, R160.l, R33
[N--D---] FOP.MOV R176.h.bf16, R33.f32
[N--D---] FOP.MUL R41, R160.l, R41
[N-sD---] FOP.MOV R177.h.bf16, R41.f32
[NwwD---]       DMA.ST.B64 R[176:177], SH[0:1], R216, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R34, R34, R192.l, R208
[N--D---] FOP.FMA R35, R35, R192.h, R209
[N--D---] FOP.FMA R42, R42, R193.l, R210
[N-sD---] FOP.FMA R43, R43, R193.h, R211
[N--D---]   FOP.CMP.LT p0, R34, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R34, R34, R232
[N--D---]   FOP.CMP.LT p0, R35, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R35, R35, R232
[N--D---]   FOP.CMP.LT p0, R42, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R42, R42, R232
[N--D---]   FOP.CMP.LT p0, R43, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N-sD---]   FOP.MUL R43, R43, R232
[N--D---] FOP.MUL R34, R161.l, R34
[N--D---] FOP.MOV R178.l.bf16, R34.f32
[N--D---] FOP.MUL R42, R161.l, R42
[N--D---] FOP.MOV R179.l.bf16, R42.f32
[N--D---] FOP.MUL R35, R161.l, R35
[N--D---] FOP.MOV R178.h.bf16, R35.f32
[N--D---] FOP.MUL R43, R161.l, R43
[N-sD---] FOP.MOV R179.h.bf16, R43.f32
[NwwD---]       DMA.ST.B64 R[178:179], SH[0:1], R217, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R36, R36, R192.l, R208
[N--D---] FOP.FMA R37, R37, R192.h, R209
[N--D---] FOP.FMA R44, R44, R193.l, R210
[N-sD---] FOP.FMA R45, R45, R193.h, R211
[N--D---]   FOP.CMP.LT p0, R36, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R36, R36, R232
[N--D---]   FOP.CMP.LT p0, R37, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R37, R37, R232
[N--D---]   FOP.CMP.LT p0, R44, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R44, R44, R232
[N--D---]   FOP.CMP.LT p0, R45, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N-sD---]   FOP.MUL R45, R45, R232
[N--D---] FOP.MUL R36, R162.l, R36
[N--D---] FOP.MOV R180.l.bf16, R36.f32
[N--D---] FOP.MUL R44, R162.l, R44
[N--D---] FOP.MOV R181.l.bf16, R44.f32
[N--D---] FOP.MUL R37, R162.l, R37
[N--D---] FOP.MOV R180.h.bf16, R37.f32
[N--D---] FOP.MUL R45, R162.l, R45
[N-sD---] FOP.MOV R181.h.bf16, R45.f32
[NwwD---]       DMA.ST.B64 R[180:181], SH[0:1], R218, _, _, _, _,  SH[0:2]
[N-wD---] FOP.FMA R38, R38, R192.l, R208
[N--D---] FOP.FMA R39, R39, R192.h, R209
[N--D---] FOP.FMA R46, R46, R193.l, R210
[N-sD---] FOP.FMA R47, R47, R193.h, R211
[N--D---]   FOP.CMP.LT p0, R38, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R38, R38, R232
[N--D---]   FOP.CMP.LT p0, R39, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R39, R39, R232
[N--D---]   FOP.CMP.LT p0, R46, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R46, R46, R232
[N--D---]   FOP.CMP.LT p0, R47, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N-sD---]   FOP.MUL R47, R47, R232
[N--D---] FOP.MUL R38, R163.l, R38
[N--D---] FOP.MOV R182.l.bf16, R38.f32
[N--D---] FOP.MUL R46, R163.l, R46
[N--D---] FOP.MOV R183.l.bf16, R46.f32
[N--D---] FOP.MUL R39, R163.l, R39
[N--D---] FOP.MOV R182.h.bf16, R39.f32
[N--D---] FOP.MUL R47, R163.l, R47
[N-sD---] FOP.MOV R183.h.bf16, R47.f32
[NswD---]       DMA.ST.B64 R[182:183], SH[0:1], R219, _, _, _, _,  SH[0:2]
[Nw-Dwww]    CTRL.NOP
[N-wD---]    CTRL.END
STORE_MASK_LEAKYRELU:
[NwwD---] FOP.FMA R32, R32, R192.l, R208
[N--D---] FOP.FMA R33, R33, R192.h, R209
[N--D---] FOP.FMA R40, R40, R193.l, R210
[N-sD---] FOP.FMA R41, R41, R193.h, R211
[N--D---]   FOP.CMP.LT p0, R32, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R32, R32, R232
[N--D---]   FOP.CMP.LT p0, R33, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R33, R33, R232
[N--D---]   FOP.CMP.LT p0, R40, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R40, R40, R232
[N--D---]   FOP.CMP.LT p0, R41, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N-sD---]   FOP.MUL R41, R41, R232
[NwwD---] FOP.MUL R32, R32, R160.l
[N--D---] FOP.MOV R32.l.bf16, R32.f32
[N--D---] FOP.MUL R33, R33, R160.l
[N--D---] FOP.MOV R33.l.bf16, R33.f32
[N--D---] FOP.MUL R40, R40, R160.l
[N--D---] FOP.MOV R40.l.bf16, R40.f32
[N--D---] FOP.MUL R41, R41, R160.l
[N-sD---] FOP.MOV R41.l.bf16, R41.f32
[NswD---]       INT.BYP.TG p0, R8.s32, C0.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R32.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R33.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R40.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R41.l, SH[0:1], R216, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R216.s32, C45.s32, R216.s32
[NwwD---] FOP.FMA R34, R34, R192.l, R208
[N--D---] FOP.FMA R35, R35, R192.h, R209
[N--D---] FOP.FMA R42, R42, R193.l, R210
[N-sD---] FOP.FMA R43, R43, R193.h, R211
[N--D---]   FOP.CMP.LT p0, R34, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R34, R34, R232
[N--D---]   FOP.CMP.LT p0, R35, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R35, R35, R232
[N--D---]   FOP.CMP.LT p0, R42, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R42, R42, R232
[N--D---]   FOP.CMP.LT p0, R43, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N-sD---]   FOP.MUL R43, R43, R232
[NwwD---] FOP.MUL R34, R34, R161.l
[N--D---] FOP.MOV R34.l.bf16, R34.f32
[N--D---] FOP.MUL R35, R35, R161.l
[N--D---] FOP.MOV R35.l.bf16, R35.f32
[N--D---] FOP.MUL R42, R42, R161.l
[N--D---] FOP.MOV R42.l.bf16, R42.f32
[N--D---] FOP.MUL R43, R43, R161.l
[N-sD---] FOP.MOV R43.l.bf16, R43.f32
[NwsD---]       INT.BYP.TG p0, R8.s32, C46.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R34.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R35.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R42.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R43.l, SH[0:1], R217, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R217.s32, C45.s32, R217.s32
[NwwD---] FOP.FMA R36, R36, R192.l, R208
[N--D---] FOP.FMA R37, R37, R192.h, R209
[N--D---] FOP.FMA R44, R44, R193.l, R210
[N-sD---] FOP.FMA R45, R45, R193.h, R211
[N--D---]   FOP.CMP.LT p0, R36, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R36, R36, R232
[N--D---]   FOP.CMP.LT p0, R37, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R37, R37, R232
[N--D---]   FOP.CMP.LT p0, R44, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R44, R44, R232
[N--D---]   FOP.CMP.LT p0, R45, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N-sD---]   FOP.MUL R45, R45, R232
[NwwD---] FOP.MUL R36, R36, R162.l
[N--D---] FOP.MOV R36.l.bf16, R36.f32
[N--D---] FOP.MUL R37, R37, R162.l
[N--D---] FOP.MOV R37.l.bf16, R37.f32
[N--D---] FOP.MUL R44, R44, R162.l
[N--D---] FOP.MOV R44.l.bf16, R44.f32
[N--D---] FOP.MUL R45, R45, R162.l
[N-sD---] FOP.MOV R45.l.bf16, R45.f32
[NswD---]       INT.BYP.TG p0, R8.s32, C39.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NswD---]   P0  DMA.ST.B16 R36.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NwsD---]   P0  DMA.ST.B16 R37.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NswD---]   P0  DMA.ST.B16 R44.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NwsD---]   P0  DMA.ST.B16 R45.l, SH[0:1], R218, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R218.s32, C45.s32, R218.s32
[NwwD---] FOP.FMA R38, R38, R192.l, R208
[N--D---] FOP.FMA R39, R39, R192.h, R209
[N--D---] FOP.FMA R46, R46, R193.l, R210
[N-sD---] FOP.FMA R47, R47, R193.h, R211
[N--D---]   FOP.CMP.LT p0, R38, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R38, R38, R232
[N--D---]   FOP.CMP.LT p0, R39, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R39, R39, R232
[N--D---]   FOP.CMP.LT p0, R46, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N--D---]   FOP.MUL R46, R46, R232
[N--D---]   FOP.CMP.LT p0, R47, C0
[N--D---]   FOP.MOVC R232, C1, SH65, p0
[N-sD---]   FOP.MUL R47, R47, R232
[NwwD---] FOP.MUL R38, R38, R163.l
[N--D---] FOP.MOV R38.l.bf16, R38.f32
[N--D---] FOP.MUL R39, R39, R163.l
[N--D---] FOP.MOV R39.l.bf16, R39.f32
[N--D---] FOP.MUL R46, R46, R163.l
[N--D---] FOP.MOV R46.l.bf16, R46.f32
[N--D---] FOP.MUL R47, R47, R163.l
[N-sD---] FOP.MOV R47.l.bf16, R47.f32
[NwsD---]       INT.BYP.TG p0, R8.s32, C48.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C0.s32
[NwsD---]   P0  DMA.ST.B16 R38.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R219.s32, C45.s32, R219.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C33.s32
[NswD---]   P0  DMA.ST.B16 R39.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R219.s32, C45.s32, R219.s32
[NswD---]       INT.BYP.TG.AP p0, R9.s32, C45.s32
[NwsD---]   P0  DMA.ST.B16 R46.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NswD---]       INT.ADD R219.s32, C45.s32, R219.s32
[NwsD---]       INT.BYP.TG.AP p0, R9.s32, C50.s32
[NswD---]   P0  DMA.ST.B16 R47.l, SH[0:1], R219, _, _, _, _,  SH[0:2]
[NwsD---]       INT.ADD R219.s32, C45.s32, R219.s32
[Nw-Dwww]    CTRL.NOP
[N-wD---]    CTRL.END


.Lfunc_end1:
	.size	musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16, .Lfunc_end1-musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16




    .type	llvm.mtgpu.kernel.musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16.sm,@object
    .local	llvm.mtgpu.kernel.musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16.sm
    .comm	llvm.mtgpu.kernel.musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16.sm,6144,4

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
      .offset:         86
      .reg_bank:       3
      .size:           80
      .values:
        - 9
        - 10
        - 11
        - 32
        - 48
        - 63
        - 64
        - 80
        - 96
        - 112
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
    .name:           musa_asm_s8mma_nhwc_conv_KRSC_128_32x64_lin_ol_outbf16
    .private_memory_size: 0
    .shared_memory_size: 6272
    .shared_reg_count: 106
    .slot_reg_count: 4
    .temp_reg_count: 238
    .wave_mode:      0
mtgpu.version:
  - 2
  - 2
...

	.mtgpu_metadata.end

    