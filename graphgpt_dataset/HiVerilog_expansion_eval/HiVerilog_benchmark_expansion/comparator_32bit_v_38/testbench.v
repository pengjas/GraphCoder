`timescale 1ns / 1ps

module tb_alu_32bit;

    // Inputs
    reg [31:0] A;
    reg [31:0] B;
    reg [1:0] op;

    // Outputs
    wire [31:0] result;
    wire carry_out;

    // Instantiate the Unit Under Test (UUT)
    alu_32bit uut (
        .A(A), 
        .B(B), 
        .op(op), 
        .result(result), 
        .carry_out(carry_out)
    );

    // Clock and Reset generation
    reg clk;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Test Cases
    integer test_number;
    initial begin
        // Initialize Inputs
        A = 0;
        B = 0;
        op = 0;
        test_number = 0;
        
        // Wait for Global Reset
        #100;
        
        // Test Case 1: Addition
        A = 32'h0001_0001;
        B = 32'h0000_0001;
        op = 2'b00;
        test_number = test_number + 1;
        #10;
        if (result != (A + B) || carry_out != 1'b0) begin
            $display("===========Error in Test Case 1: Addition===========");
            $finish;
        end
        
        // Test Case 2: Subtraction
        A = 32'h0001_0000;
        B = 32'h0000_0001;
        op = 2'b01;
        test_number = test_number + 1;
        #10;
        if (result != (A - B) || carry_out != 1'b0) begin
            $display("===========Error in Test Case 2: Subtraction===========");
            $finish;
        end
        
        // Test Case 3: AND operation
        A = 32'hFFFF_FFFF;
        B = 32'h0000_FFFF;
        op = 2'b10;
        test_number = test_number + 1;
        #10;
        if (result != (A & B) || carry_out != 1'b0) begin
            $display("===========Error in Test Case 3: AND Operation===========");
            $finish;
        end

        // All test cases passed
        $display("===========Your Design Passed===========");
        $finish;
    end
      
endmodule
