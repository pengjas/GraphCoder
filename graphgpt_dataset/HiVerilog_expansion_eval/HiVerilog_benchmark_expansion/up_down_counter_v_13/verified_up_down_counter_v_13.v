module configurable_alu (
    input wire clk,
    input wire [1:0] op_code,
    input wire [31:0] operand_a,
    input wire [31:0] operand_b,
    output wire [31:0] result
);

    wire [31:0] add_result;
    wire [31:0] sub_result;
    wire [31:0] and_or_result;

    // Instantiate adder submodule
    adder u_adder (
        .operand_a(operand_a),
        .operand_b(operand_b),
        .result(add_result)
    );

    // Instantiate subtractor submodule
    subtractor u_subtractor (
        .operand_a(operand_a),
        .operand_b(operand_b),
        .result(sub_result)
    );

    // Instantiate bitwise operator submodule
    bitwise_operator u_bitwise_operator (
        .op_code(op_code[0]),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .result(and_or_result)
    );

    // Result logic based on operation code
    assign result = (op_code == 2'b00) ? add_result :
                    (op_code == 2'b01) ? sub_result :
                    and_or_result;

endmodule

module adder (
    input wire [31:0] operand_a,
    input wire [31:0] operand_b,
    output wire [31:0] result
);
    assign result = operand_a + operand_b;
endmodule

module subtractor (
    input wire [31:0] operand_a,
    input wire [31:0] operand_b,
    output wire [31:0] result
);
    assign result = operand_a - operand_b;
endmodule

module bitwise_operator (
    input wire op_code,
    input wire [31:0] operand_a,
    input wire [31:0] operand_b,
    output wire [31:0] result
);
    assign result = (op_code) ? (operand_a | operand_b) : (operand_a & operand_b);
endmodule
