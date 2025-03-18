module simple_alu(
    input [3:0] a,
    input [3:0] b,
    input [1:0] op,
    output reg [3:0] result
);
wire [3:0] sum;
wire [3:0] diff;
wire [3:0] and_res;
wire [3:0] or_res;

adder_module adder(.a(a), .b(b), .sum(sum));
subtractor_module subtractor(.a(a), .b(b), .diff(diff));
and_module and_op(.a(a), .b(b), .and_res(and_res));
or_module or_op(.a(a), .b(b), .or_res(or_res));

always @(*) begin
    case(op)
        2'b00: result = sum;
        2'b01: result = diff;
        2'b10: result = and_res;
        2'b11: result = or_res;
    endcase
end
endmodule

module adder_module(
    input [3:0] a,
    input [3:0] b,
    output [3:0] sum
);
assign sum = a + b;
endmodule

module subtractor_module(
    input [3:0] a,
    input [3:0] b,
    output [3:0] diff
);
assign diff = a - b;
endmodule

module and_module(
    input [3:0] a,
    input [3:0] b,
    output [3:0] and_res
);
assign and_res = a & b;
endmodule

module or_module(
    input [3:0] a,
    input [3:0] b,
    output [3:0] or_res
);
assign or_res = a | b;
endmodule