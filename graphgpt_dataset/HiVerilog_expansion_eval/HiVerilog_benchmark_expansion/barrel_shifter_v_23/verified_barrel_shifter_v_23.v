module simple_alu(a, b, op, result, carry_borrow);
    input [3:0] a, b;
    input [1:0] op;
    output [3:0] result;
    output carry_borrow;
    wire [3:0] w_and, w_or, w_add, w_sub;
    wire c_add, b_sub;

    bitwise_and and_unit(a, b, w_and);
    bitwise_or or_unit(a, b, w_or);
    adder_4bit add_unit(a, b, w_add, c_add);
    subtractor_4bit sub_unit(a, b, w_sub, b_sub);

    assign result = (op == 2'b00) ? w_and :
                    (op == 2'b01) ? w_or :
                    (op == 2'b10) ? w_add :
                    w_sub;
                    
    assign carry_borrow = (op == 2'b10) ? c_add : b_sub;

endmodule

module bitwise_and(input [3:0] a, input [3:0] b, output [3:0] out);
    assign out = a & b;
endmodule

module bitwise_or(input [3:0] a, input [3:0] b, output [3:0] out);
    assign out = a | b;
endmodule

module adder_4bit(input [3:0] a, input [3:0] b, output [3:0] sum, output carry);
    assign {carry, sum} = a + b;
endmodule

module subtractor_4bit(input [3:0] a, input [3:0] b, output [3:0] diff, output borrow);
    assign {borrow, diff} = {1'b0, a} - {1'b0, b};
endmodule
