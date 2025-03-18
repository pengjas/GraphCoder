`timescale 1ns / 1ps

module tb_dual_shift_register;

    // Inputs
    reg clk;
    reg rst;
    reg load0;
    reg load1;
    reg [7:0] data0;
    reg [7:0] data1;
    reg shift0;
    reg shift1;

    // Outputs
    wire [7:0] out0;
    wire [7:0] out1;

    // Instantiate the Unit Under Test (UUT)
    dual_shift_register uut (
        .clk(clk),
        .rst(rst),
        .load0(load0),
        .load1(load1),
        .data0(data0),
        .data1(data1),
        .shift0(shift0),
        .shift1(shift1),
        .out0(out0),
        .out1(out1)
    );

    // Clock generation
    always #5 clk = ~clk;  // 100 MHz Clock

    // Test cases
    initial begin
        // Initialize Inputs
        clk = 0;
        rst = 1;
        load0 = 0;
        load1 = 0;
        data0 = 0;
        data1 = 0;
        shift0 = 0;
        shift1 = 0;

        // Reset pulse
        #10;
        rst = 0;
        #10;
        rst = 1;
        #10;
        rst = 0;
        
        // Test case 1: Load data into both shift registers
        load0 = 1;
        load1 = 1;
        data0 = 8'hAA;  // Load data 10101010
        data1 = 8'h55;  // Load data 01010101
        #10;
        load0 = 0;
        load1 = 0;

        // Test case 2: Shift both registers to the right
        shift0 = 1;
        shift1 = 1;
        #10;
        shift0 = 0;
        shift1 = 0;

        // Check results after shifting
        if (out0 !== 8'h55 || out1 !== 8'h2A) begin
            $display("===========Error===========");
            $stop;
        end

        // Additional test cases and shift operations
        // Adding multiple shifts and checks here...

        // Final check and successful completion message
        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
