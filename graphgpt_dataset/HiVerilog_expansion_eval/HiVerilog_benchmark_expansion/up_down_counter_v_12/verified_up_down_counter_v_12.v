module multi_function_alu (
    input wire [2:0] op_code,
    input wire [31:0] operand_a,
    input wire [31:0] operand_b,
    output wire [31:0] result,
    output wire zero
);

    wire [31:0] sum, difference, and_res, or_res, xor_res;
    reg [31:0] alu_result;

    // Instantiate adder
    adder u_adder (
        .a(operand_a),
        .b(operand_b),
        .sum(sum)
    );

    // Instantiate subtractor
    subtractor u_subtractor (
        .a(operand_a),
        .b(operand_b),
        .difference(difference)
    );

    // Instantiate AND logic
    and_bit u_and_bit (
        .a(operand_a),
        .b(operand_b),
        .and_res(and_res)
    );

    // Instantiate OR logic
    or_bit u_or_bit (
        .a(operand_a),
        .b(operand_b),
        .or_res(or_res)
    );

    // Instantiate XOR logic
    xor_bit u_xor_bit (
        .a(operand_a),
        .b(operand_b),
        .xor_res(xor_res)
    );

    // Result selection logic
    always @(*) begin
        case(op_code)
            3'b000: alu_result = sum;
            3'b001: alu_result = difference;
            3'b010: alu_result = and_res;
            3'b011: alu_result = or_res;
            3'b100: alu_result = xor_res;
            default: alu_result = 32'b0;
        endcase
    end

    assign result = alu_result;
    assign zero = (result == 32'b0);

endmodule

module adder (
    input wire [31:0] a,
    input wire [31:0] b,
    output wire [31:0] sum
);
    assign sum = a + b;
endmodule

module subtractor (
    input wire [31:0] a,
    input wire [31:0] b,
    output wire [31:0] difference
);
    assign difference = a - b;
endmodule

module and_bit (
    input wire [31:0] a,
    input wire [31:0] b,
    output wire [31:0] and_res
);
    assign and_res = a & b;
endmodule

module or_bit (
    input wire [31:0] a,
    input wire [31:0] b,
    output wire [31:0] or_res
);
    assign or_res = a | b;
endmodule

module xor_bit (
    input wire [31:0] a,
    input wire [31:0] b,
    output wire [31:0] xor_res
);
    assign xor_res = a ^ b;
endmodule
