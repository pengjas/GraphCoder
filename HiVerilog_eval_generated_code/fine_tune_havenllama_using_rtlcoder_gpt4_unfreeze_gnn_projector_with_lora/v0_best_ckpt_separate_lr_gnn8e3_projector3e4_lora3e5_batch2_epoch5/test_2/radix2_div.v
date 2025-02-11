module radix2_div(
    input wire clk,
    input wire rst,
    input wire [7:0] dividend,    
    input wire [7:0] divisor,    
    input wire sign,       
    input wire opn_valid,   
    output reg res_valid,   
    input wire res_ready,   
    output wire [15:0] result
);

wire [15:0] abs_dividend;
wire [15:0] abs_divisor;
wire [15:0] quotient;
wire [15:0] remainder;

wire [3:0] counter;
wire [3:0] shift_reg;
wire [15:0] shift_result;

wire [15:0] abs_divisor_mod;
wire [15:0] neg_divisor_mod;
wire [15:0] abs_dividend_mod;
wire [15:0] abs_divisor_neg_mod;

wire [15:0] ctrl_result;
wire [15:0] ctrl_remainder;

wire [15:0] res_mgmt;

wire [15:0] abs_dividend_mod_res_mgmt;
wire [15:0] abs_divisor_mod_res_mgmt;
wire [15:0] neg_divisor_mod_res_mgmt;
wire [15:0] abs_divisor_neg_mod_res_mgmt;

wire [15:0] ctrl_quotient;
wire [15:0] ctrl_remainder_res_mgmt;

wire [15:0] ctrl_quotient_res_mgmt;
wire [15:0] ctrl_remainder_res_mgmt;

wire [15:0] ctrl_quotient_res_mgmt_res_mgmt;
wire [15:0] ctrl_remainder_res_mgmt_res_mgmt;
endmodule