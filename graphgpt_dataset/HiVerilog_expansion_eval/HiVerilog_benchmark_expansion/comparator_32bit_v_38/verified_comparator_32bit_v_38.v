module alu_32bit(
    input [31:0] A,
    input [31:0] B,
    input [1:0] op,
    output [31:0] result,
    output carry_out
);

    wire [31:0] sum;
    wire [31:0] diff;
    wire [31:0] and_result;
    wire sum_carry_out;
    wire diff_borrow_out;

    // Instantiate the adder module
    adder_32bit add_inst (
        .A(A),
        .B(B),
        .sum(sum),
        .carry_out(sum_carry_out)
    );

    // Instantiate the subtractor module
    subtractor_32bit sub_inst (
        .A(A),
        .B(B),
        .diff(diff),
        .borrow_out(diff_borrow_out)
    );

    // Instantiate the AND module
    and_32bit and_inst (
        .A(A),
        .B(B),
        .and_result(and_result)
    );

    // Multiplexer for selecting operation result
    assign result = (op == 2'b00) ? sum :
                    (op == 2'b01) ? diff :
                    (op == 2'b10) ? and_result : 32'bx;

    // Multiplexer for carry output
    assign carry_out = (op == 2'b00) ? sum_carry_out :
                       (op == 2'b01) ? diff_borrow_out : 1'b0;

endmodule

module adder_32bit(
    input [31:0] A,
    input [31:0] B,
    output [31:0] sum,
    output carry_out
);
    assign {carry_out, sum} = A + B;
endmodule

module subtractor_32bit(
    input [31:0] A,
    input [31:0] B,
    output [31:0] diff,
    output borrow_out
);
    assign {borrow_out, diff} = A - B;
endmodule

module and_32bit(
    input [31:0] A,
    input [31:0] B,
    output [31:0] and_result
);
    assign and_result = A & B;
endmodule
