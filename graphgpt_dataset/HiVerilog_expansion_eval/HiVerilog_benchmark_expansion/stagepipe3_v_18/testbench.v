`timescale 1ns / 1ps

module tb_simple_alu4bit;

    reg clk;
    reg [1:0] op_code;
    reg [3:0] operand_a;
    reg [3:0] operand_b;
    wire [3:0] result;
    reg [3:0] expected_result;
    reg [15:0] test_vector [0:15]; // Storage for test vectors
    integer i;
    reg test_failed;

    // Instantiate the Unit Under Test (UUT)
    simple_alu4bit uut (
        .clk(clk),
        .op_code(op_code),
        .operand_a(operand_a),
        .operand_b(operand_b),
        .result(result)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Initialize and load test vectors
    initial begin
        // Initialize signals
        clk = 0;
        op_code = 0;
        operand_a = 0;
        operand_b = 0;
        test_failed = 0;

        // Load test vectors
        // Each test vector: {op_code, operand_a, operand_b, expected_result}
        test_vector[0] = {2'b00, 4'd3, 4'd2, 4'd5}; // Add
        test_vector[1] = {2'b01, 4'd5, 4'd3, 4'd2}; // Subtract
        test_vector[2] = {2'b10, 4'd5, 4'd3, 4'd1}; // AND
        test_vector[3] = {2'b11, 4'd5, 4'd3, 4'd7}; // OR
        test_vector[4] = {2'b00, 4'd15, 4'd1, 4'd0}; // Add overflow
        test_vector[5] = {2'b01, 4'd0, 4'd1, 4'd15}; // Subtract underflow
        test_vector[6] = {2'b10, 4'd0, 4'd1, 4'd0}; // AND zero
        test_vector[7] = {2'b11, 4'd15, 4'd0, 4'd15}; // OR one-side zero
        
        // More test cases...

        // Begin testing
        for (i = 0; i < 8; i = i + 1) begin
            {op_code, operand_a, operand_b, expected_result} = test_vector[i];
            #10; // wait for the next clock edge
            if (result !== expected_result) begin
                $display("Test %d failed: op_code=%b operand_a=%d operand_b=%d Expected=%d Got=%d", 
                         i, op_code, operand_a, operand_b, expected_result, result);
                test_failed = 1;
            end
        end

        // Test result reporting
        if (test_failed == 0) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error===========");
        end
        
        $finish;
    end

endmodule
