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
wire [7:0] shift_reg;
wire [3:0] counter;
wire [15:0] abs_divisor_neg;
wire [15:0] abs_dividend_neg;
wire [15:0] abs_dividend_neg_divisor;
wire [15:0] abs_dividend_neg_divisor_shift;
wire [15:0] abs_dividend_neg_divisor_shift_neg;
wire [15:0] abs_dividend_neg_divisor_shift_neg_quotient;
wire [15:0] abs_dividend_neg_divisor_shift_neg_remainder;
endmodule