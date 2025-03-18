`timescale 1ns / 1ps

module tb_adder_selector;

    reg [7:0] a;
    reg [7:0] b;
    reg ctrl;
    reg clk;
    reg rst;
    wire [7:0] sum;

    // Instantiate the Unit Under Test (UUT)
    adder_selector uut (
        .a(a),
        .b(b),
        .ctrl(ctrl),
        .sum(sum)
    );

    // Clock generation
    always #5 clk = ~clk;

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
        ctrl = 0;
        clk = 0;

        // Wait for Global Reset
        @(negedge rst);
        #10;
        
        // Test Case 1: Check zeroing functionality
        a = 8'h55;  // 85
        b = 8'hAA;  // 170
        ctrl = 1'b0; // Output should be zero
        #10;
        if (sum !== 8'h00) begin
            $display("===========Error in Test Case 1: sum=%h, expected=%h===========", sum, 8'h00);
            $finish;
        end
        
        // Test Case 2: Check addition functionality
        ctrl = 1'b1; // Output should be a + b
        #10;
        if (sum !== (a + b)) begin
            $display("===========Error in Test Case 2: sum=%h, expected=%h===========", sum, (a + b));
            $finish;
        end

        // Test Case 3: Another check for zeroing with different values
        a = 8'hFF;  // 255
        b = 8'h01;  // 1
        ctrl = 1'b0;
        #10;
        if (sum !== 8'h00) begin
            $display("===========Error in Test Case 3: sum=%h, expected=%h===========", sum, 8'h00);
            $finish;
        end

        // Test Case 4: Edge case for addition (overflow)
        a = 8'hFF;
        b = 8'h02;
        ctrl = 1'b1;
        #10;
        if (sum !== (a + b)) begin
            $display("===========Error in Test Case 4: sum=%h, expected=%h===========", sum, (a + b));
            $finish;
        end

        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
