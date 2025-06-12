`timescale 1ns / 1ps

module tb_subtractor_32bit;

    // Inputs
    reg [31:0] A;
    reg [31:0] B;
    reg Bin;
    
    // Outputs
    wire [31:0] D;
    wire Bout;
    
    // Instantiate the Unit Under Test (UUT)
    subtractor_32bit uut (
        .A(A), 
        .B(B), 
        .Bin(Bin), 
        .D(D), 
        .Bout(Bout)
    );

    // Clock and Reset Generation
    reg clk = 0;
    always #5 clk = !clk; // Clock with period 10ns

    // Test Cases
    initial begin
        // Initialize Inputs
        A = 0; B = 0; Bin = 0;

        // Wait for global reset to finish
        #100;
        
        // Test Case 1: Simple subtraction without borrow
        A = 32'h0000_0001; B = 32'h0000_0001; Bin = 1'b0;
        #10;
        if ((D !== 32'h0000_0000) || (Bout !== 1'b0)) begin
            $display("===========Error in Test Case 1===========");
            $finish;
        end
        
        // Test Case 2: Subtraction with borrow
        A = 32'h0000_0000; B = 32'h0000_0001; Bin = 1'b0;
        #10;
        if ((D !== 32'hFFFF_FFFF) || (Bout !== 1'b1)) begin
            $display("===========Error in Test Case 2===========");
            $finish;
        end
        
        // Test Case 3: Large numbers subtraction
        A = 32'hFFFF_0000; B = 32'h0000_FFFF; Bin = 1'b0;
        #10;
        if ((D !== 32'hFFFE_0001) || (Bout !== 1'b0)) begin
            $display("===========Error in Test Case 3===========");
            $finish;
        end
        
        // Test Case 4: Check zero borrow out with equal numbers
        A = 32'h1234_5678; B = 32'h1234_5678; Bin = 1'b0;
        #10;
        if ((D !== 32'h0000_0000) || (Bout !== 1'b0)) begin
            $display("===========Error in Test Case 4===========");
            $finish;
        end
        
        // All tests passed
        $display("===========Your Design Passed===========");
        $finish;
    end
    
endmodule
