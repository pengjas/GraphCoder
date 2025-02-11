module alu(
    input [31:0] a,
    input [31:0] b,
    input [5:0] aluc,
    output [31:0] r,
    output zero,
    output carry,
    output negative,
    output overflow,
    output flag
);

wire [31:0] add_result;
wire [31:0] sub_result;
wire [31:0] and_result;
wire [31:0] or_result;
wire [31:0] xor_result;
wire [31:0] nor_result;
wire [31:0] slt_result;
wire [31:0] sltu_result;
wire [31:0] sll_result;
wire [31:0] srl_result;
wire [31:0] sra_result;
wire [31:0] sllv_result;
wire [31:0] srlv_result;
wire [31:0] srav_result;
wire [31:0] lui_result;



wire [31:0] a_add_b;
wire [31:0] b_add_a;



wire [31:0] a_sub_b;
wire [31:0] b_sub_a;



wire [31:0] a_and_b;
wire [31:0] a_or_b;
wire [31:0] a_xor_b;
wire [31:0] a_nor_b;
wire [31:0] a_slt_b;
wire [31:0] a_sltu_b;
wire [31:0] a_sll_b;
wire [31:0] a_srl_b;
wire [31:0] a_srav_b;
wire [31:0] a_sllv_b;



wire [31:0] a_slt_b_flag;
wire [31:0] a_sltu_b_flag;
wire [31:0] a_sll_b_flag;
wire [31:0] a_srl_b_flag;
wire [31:0] a_srav_b_flag;
wire [31:0] a_sllv_b_flag;



assign a_add_b = a + b;
assign b_add_a = b + a;



assign a_sub_b = a - b;
assign b_sub_a = b - a;



assign and_result = a & b;
assign or_result = a | b;
assign xor_result = a ^ b;
assign nor_result = a ~ b;
assign slt_result = a < b;
assign sltu_result = a < b;
assign sll_result = a << b;
assign srl_result = a >> b;
assign srav_result = a >>> b;



assign a_slt_b_flag = (slt_result == 0) ? 1'b1 : 1'b0;
assign a_sltu_b_flag = (sltu_result == 0) ? 1'b1 : 1'b0;
assign a_sll_b_flag = (sll_result == 0) ? 1'b1 : 1'b0;
assign a_srl_b_flag = (srl_result == 0) ? 1'b1 : 1'b0;
assign a_srav_b_flag = (srav_result == 0) ? 1'b1 : 1'b0;


assign r = (aluc == 6'b10000) ? a_add_b :
                  (aluc == 6'b10001) ? b_add_a :
                  (aluc == 6'b100010) ? a_sub_b :
endmodule