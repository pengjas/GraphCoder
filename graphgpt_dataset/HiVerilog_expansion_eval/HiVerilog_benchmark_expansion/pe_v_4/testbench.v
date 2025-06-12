`timescale 1ns / 1ps

module tb_dual_mode_arithmetic;

    // Inputs
    reg clk;
    reg rst;
    reg mode;
    reg [15:0] a;
    reg [15:0] b;

    // Outputs
    wire [15:0] result;

    // Instantiate the Unit Under Test (UUT)
    dual_mode_arithmetic uut (
        .clk(clk),
        .rst(rst),
        .mode(mode),
        .a(a),
        .b(b),
        .result(result)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = !clk; // Clock period of 10ns
    end

    // Reset generation
    initial begin
        rst = 1;
        #15;
        rst = 0;
    end

    // Test cases
    initial begin
        // Initialize Inputs
        a = 0;
        b = 0;
        mode = 0;

        // Wait for reset to finish
        #20;
        
        // Test case 1: Addition 1000 + 500
        a = 1000;
        b = 500;
        mode = 0; // Addition
        #10;
        if (result !== 1500) begin
            $display("===========Error: Addition Test Failed (1000 + 500)===========");
            $finish;
        end

        // Test case 2: Subtraction 1000 - 500
        mode = 1; // Subtraction
        #10;
        if (result !== 500) begin
            $display("===========Error: Subtraction Test Failed (1000 - 500)===========");
            $finish;
        end
        
        // Test case 3: Addition with overflow
        a = 16'h8000; // -32768 in signed
        b = 16'h8000; // -32768 in signed
        mode = 0; // Addition
        #10;
        if (result !== 16'h0000) begin
            $display("===========Error: Addition Overflow Test Failed===========");
            $finish;
        end

        // Test case 4: Subtraction to negative
        a = 500;
        b = 1000;
        mode = 1; // Subtraction
        #10;
        if (result !== 16'hFE0C) begin // -500 in hex
            $display("===========Error: Subtraction Negative Result Test Failed===========");
            $finish;
        end

        // All tests passed
        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
