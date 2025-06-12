`timescale 1ns / 1ps

module multiplier(
    input [31:0] a,
    input [31:0] b,
    input multc,
    output [63:0] y,
    output [31:0] msb,
    output [31:0] lsb,
    output zero
);

wire [63:0] mult_signed_result, mult_unsigned_result;

// Instantiate multiplication operations
signed_mult mult_signed(.a(a), .b(b), .result(mult_signed_result));
unsigned_mult mult_unsigned(.a(a), .b(b), .result(mult_unsigned_result));

// Result selection based on multc
assign y = (multc == 1'b0) ? mult_signed_result : mult_unsigned_result;
assign msb = y[63:32];
assign lsb = y[31:0];
assign zero = (y == 64'b0);

endmodule

// Submodules definition
module signed_mult(
    input [31:0] a,
    input [31:0] b,
    output [63:0] result
);
    assign result = $signed(a) * $signed(b);
endmodule

module unsigned_mult(
    input [31:0] a,
    input [31:0] b,
    output [63:0] result
);
    assign result = a * b;
endmodule
