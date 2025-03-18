`timescale 1ns / 1ps

module tb_simple_alu;

    // Inputs
    reg [3:0] a;
    reg [3:0] b;
    reg [1:0] op;
    reg clk;
    reg rst;

    // Outputs
    wire [3:0] result;
    wire carry_borrow;

    // Instantiate the Unit Under Test (UUT)
    simple_alu uut (
        .a(a), 
        .b(b), 
        .op(op), 
        .result(result), 
        .carry_borrow(carry_borrow)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Initial block and test vectors
    initial begin
        // Initialize Inputs
        clk = 0;
        rst = 1;
        a = 0;
        b = 0;
        op = 0;

        // Reset
        #10;
        rst = 0;
        #10;
        rst = 1;
        #10;
        
        // Test Case 1: AND operation
        a = 4'b1101;
        b = 4'b1011;
        op = 2'b00; // AND
        #10;
        check_result(4'b1001, 1'b0);

        // Test Case 2: OR operation
        a = 4'b1101;
        b = 4'b1011;
        op = 2'b01; // OR
        #10;
        check_result(4'b1111, 1'b0);

        // Test Case 3: Addition
        a = 4'b0101;
        b = 4'b0011;
        op = 2'b10; // Addition
        #10;
        check_result(4'b1000, 1'b0);

        // Test Case 4: Subtraction
        a = 4'b1001;
        b = 4'b0011;
        op = 2'b11; // Subtraction
        #10;
        check_result(4'b0110, 1'b0);

        // Test Case 5: Subtraction with Borrow
        a = 4'b0010;
        b = 4'b0100;
        op = 2'b11; // Subtraction
        #10;
        check_result(4'b1110, 1'b1);

        // All tests done, check for success
        #10;
        $display("===========Your Design Passed===========");
        $finish;
    end

    task check_result;
        input [3:0] expected_result;
        input expected_carry_borrow;
        begin
            if(result !== expected_result || carry_borrow !== expected_carry_borrow) begin
                $display("===========Error===========");
                $display("Failure at op=%b: Expected result=%b, carry_borrow=%b; Received result=%b, carry_borrow=%b", op, expected_result, expected_carry_borrow, result, carry_borrow);
                $finish;
            end
        end
    endtask

endmodule
