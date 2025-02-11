module barrel_shifter (in, ctrl, out);
input wire [7:0] in;
    input wire [2:0] ctrl;
    output wire [7:0] out;

    wire [7:0] stage1_out;
    wire [7:0] stage2_out;
    wire [7:0] stage3_out;
    wire [7:0] stage4_out;
    wire [7:0] stage5_out;
    wire [7:0] stage6_out;

    wire [7:0] mux2X1_out;

    assign stage1_out = {in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]};
    assign stage2_out = {stage1_out[0], stage1_out[1], stage1_out[2], stage1_out[3], stage1_out[4], stage1_out[5], stage1_out[6], stage1_out[7]};
    assign stage3_out = {stage2_out[0], stage2_out[1], stage2_out[2], stage2_out[3], stage2_out[4], stage2_out[5], stage2_out[6], stage2_out[7]};
    assign stage4_out = {stage3_out[0], stage3_out[1], stage3_out[2], stage3_out[3], stage3_out[4], stage3_out[5], stage3_out[6], stage3_out[7]};
    assign stage5_out = {stage4_out[0], stage4_out[1], stage4_out[2], stage4_out[3], stage4_out[4], stage4_out[5], stage4_out[6], stage4_out[7]};
    assign stage6_out = {stage5_out[0], stage5_out[1], stage5_out[2], stage5_out[3], stage5_out[4], stage5_out[5], stage5_out[6], stage5_out[7]};

    assign mux2X1_out = {stage6_out[0], stage6_out[1], stage6_out[2], stage6_out[3], stage6_out[4], stage6_out[5], stage6_out[6], stage6_out[7]};

    assign out = mux2X1_out;

endmodule


module barrel_shifter (
    input wire [7:0] in,
    input wire [2:0] ctrl,
    output wire [7:0] out
);

    wire [7:0] stage1_out;
    wire [7:0] stage2_out;
    wire [7:0] stage3_out;
    wire [7:0] stage4_out;
    wire [7:0] stage5_out;
    wire [7:0] stage6_out;

    wire [7:0] mux2X1_out;

    assign stage1_out = {in[0], in[1], in[2], in[3], in[4], in[5], in[6], in[7]};
    assign stage2_out = {stage1_out[0], stage1_out[1], stage1_out[2], stage1_out[3], stage1_out[4], stage1_out[5], stage1_out[6], stage1_out[7]};
    assign stage3_out = {stage2_out[0], stage2_out[1], stage2_out[2], stage2_out[3], stage2_out[4], stage2_out[5], stage2_out[6], stage2_out[7]};
    assign stage4_out = {stage3_out[0], stage3_out[1], stage3_out[2], stage3_out[3], stage3_out[4], stage3_out[5], stage3_out[6], stage3_out[7]};
    assign stage5_out = {stage4_out[0], stage4_out[1], stage4_out[2], stage4_out[3], stage4_out[4], stage4_out[5
endmodule