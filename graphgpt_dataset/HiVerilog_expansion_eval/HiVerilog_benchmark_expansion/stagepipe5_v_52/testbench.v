`timescale 1ns / 1ps

module tb_simple_alu;
    // Inputs
    reg clk;
    reg rst;
    reg [1:0] op_code;
    reg [31:0] operand_a;
    reg [31:0] operand_b;

    // Outputs
    wire [31:0] result;

    // Instantiate the Unit Under Test (UUT)
    simple_alu uut (
        .clk(clk),
        .rst(rst),
        .op_code(op_code),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .result(result)
    );

    // Clock generation
    always #10 clk = ~clk;

    // Stimuli application and result checking
    initial begin
        // Initialize Inputs
        clk = 0;
        rst = 1;
        op_code = 0;
        operand_a = 0;
        operand_b = 0;

        // Reset the UUT
        #20;
        rst = 0;

        // Test Addition
        op_code = 2'b00;
        operand_a = 32'd15;
        operand_b = 32'd10;
        #20;
        if (result !== 32'd25) $display("Error in addition: %d", result);
        
        // Test Subtraction
        op_code = 2'b01;
        operand_a = 32'd25;
        operand_b = 32'd10;
        #20;
        if (result !== 32'd15) $display("Error in subtraction: %d", result);

        // Test Bitwise AND
        op_code = 2'b10;
        operand_a = 32'd12; // 1100
        operand_b = 32'd10; // 1010
        #20;
        if (result !== 32'd8) $display("Error in AND operation: %d", result); // 1000

        // Test Bitwise OR
        op_code = 2'b11;
        operand_a = 32'd12; // 1100
        operand_b = 32'd10; // 1010
        #20;
        if (result !== 32'd14) $display("Error in OR operation: %d", result); // 1110

        // All tests passed
        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
