`timescale 1ns / 1ps

module test_dual_latch_system;

    // Inputs
    reg clk;
    reg reset;
    reg [7:0] d0;
    reg [7:0] d1;
    reg load0;
    reg load1;

    // Outputs
    wire [7:0] q0;
    wire [7:0] q1;

    // Instantiate the Unit Under Test (UUT)
    dual_latch_system uut (
        .clk(clk), 
        .reset(reset), 
        .d0(d0), 
        .d1(d1), 
        .load0(load0), 
        .load1(load1), 
        .q0(q0), 
        .q1(q1)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #10 clk = ~clk; // Clock period of 20ns
    end

    // Test scenarios
    initial begin
        // Initialize inputs
        reset = 1;
        d0 = 0;
        d1 = 0;
        load0 = 0;
        load1 = 0;

        // Wait for global reset
        #100;
        reset = 0;
        
        // Test Case 1: Load d0 into q0
        d0 = 8'hAA;  // Assign a pattern
        load0 = 1;   // Load it
        #20;
        load0 = 0;   // Remove the load signal
        if (q0 !== 8'hAA) begin
            $display("===========Error in Test Case 1: q0 = %h, Expected = %h===========", q0, 8'hAA);
            $finish;
        end
        
        // Test Case 2: Load d1 into q1
        d1 = 8'h55;  // Assign a pattern
        load1 = 1;   // Load it
        #20;
        load1 = 0;   // Remove the load signal
        if (q1 !== 8'h55) begin
            $display("===========Error in Test Case 2: q1 = %h, Expected = %h===========", q1, 8'h55);
            $finish;
        end

        // Test Case 3: Check reset behavior
        reset = 1;
        #20;
        if (q0 !== 8'h00 || q1 !== 8'h00) begin
            $display("===========Error in Test Case 3: q0 = %h or q1 = %h not reset===========", q0, q1);
            $finish;
        end
        reset = 0;

        $display("===========Your Design Passed===========");
        $finish;
    end

endmodule
