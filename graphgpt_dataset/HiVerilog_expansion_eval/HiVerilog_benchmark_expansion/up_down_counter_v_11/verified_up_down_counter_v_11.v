module modular_alu (
    input wire [1:0] op_code,
    input wire [31:0] a,
    input wire [31:0] b,
    output wire [31:0] result,
    output wire zero
);

    wire [31:0] sum_result;
    wire [31:0] sub_result;
    wire [31:0] and_result;
    wire [31:0] or_result;

    // Instantiate the adder submodule
    adder u_adder (
        .a(a),
        .b(b),
        .sum(sum_result)
    );

    // Instantiate the subtractor submodule
    subtractor u_subtractor (
        .a(a),
        .b(b),
        .sub(sub_result)
    );

    // Instantiate the AND logic submodule
    and_logic u_and_logic (
        .a(a),
        .b(b),
        .and_out(and_result)
    );

    // Instantiate the OR logic submodule
    or_logic u_or_logic (
        .a(a),
        .b(b),
        .or_out(or_result)
    );

    // Logic to select the output
    assign result = (op_code == 2'b00) ? sum_result :
                    (op_code == 2'b01) ? sub_result :
                    (op_code == 2'b10) ? and_result :
                    or_result;

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
    output wire [31:0] sub
);
    assign sub = a - b;
endmodule

module and_logic (
    input wire [31:0] a,
    input wire [31:0] b,
    output wire [31:0] and_out
);
    assign and_out = a & b;
endmodule

module or_logic (
    input wire [31:0] a,
    input wire [31:0] b,
    output wire [31:0] or_out
);
    assign or_out = a | b;
endmodule
