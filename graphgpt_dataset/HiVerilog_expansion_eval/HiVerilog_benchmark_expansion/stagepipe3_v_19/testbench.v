`timescale 1ns / 1ps

module tb_simple_alu;

    // Inputs
    reg [3:0] a;
    reg [3:0] b;
    reg [1:0] op;
    reg clk;
    reg reset;

    // Outputs
    wire [3:0] result;

    // Instantiate the Unit Under Test (UUT)
    simple_alu uut (
        .a(a), 
        .b(b), 
        .op(op), 
        .result(result)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = !clk;
    end

    // Reset generation
    initial begin
        reset = 1;
        #10;
        reset = 0;
    end

    // Stimuli and Checking
    initial begin
        // Initialize Inputs
        a = 0;
        b = 0;
        op = 0;

        // Wait for reset deassertion
        wait (reset == 0);
        #10;

        // Test Case 1: Addition
        a = 4'd3;
        b = 4'd2;
        op = 2'b00; // ADD
        #10;
        if (result != 4'd5) begin
            $display("===========Error in ADD===========");
            $finish;
        end
        
        // Test Case 2: Subtraction
        a = 4'd7;
        b = 4'd4;
        op = 2'b01; // SUBTRACT
        #10;
        if (result != 4'd3) begin
            $display("===========Error in SUBTRACT===========");
            $finish;
        end

        // Test Case 3: Bitwise AND
        a = 4'd12; // 1100
        b = 4'd10; // 1010
        op = 2'b10; // AND
        #10;
        if (result != 4'd8) begin // 1000
            $display("===========Error in AND===========");
            $finish;
        end

        // Test Case 4: Bitwise OR
        a = 4'd9;  // 1001
        b = 4'd4;  // 0100
        op = 2'b11; // OR
        #10;
        if (result != 4'd13) begin // 1101
            $display("===========Error in OR===========");
            $finish;
        end
        
        $display("===========Your Design Passed===========");
        $finish;
    end
    
endmodule
