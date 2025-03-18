`timescale 1ns / 1ps

module testbench;

    // Inputs
    reg clk;
    reg control_signal;

    // Outputs
    wire [31:0] count;

    // Instantiate the Unit Under Test (UUT)
    binary_ripple_counter uut (
        .clk(clk),
        .control_signal(control_signal),
        .count(count)
    );

    // Clock generation
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // Clock period of 10ns
    end

    // Variable to monitor test status
    reg test_passed;

    // Initial setup and tests
    initial begin
        // Initialize Inputs
        control_signal = 0;
        test_passed = 1;

        // Wait for global reset
        #(20);

        // Test Case 1: Check if the counter resets
        control_signal = 0;
        #(10); // Wait for a few clock cycles
        if (count !== 32'b0) begin
            $display("Test Case 1 Failed: Counter did not reset. count = %d", count);
            test_passed = 0;
        end

        // Test Case 2: Check counter increment
        control_signal = 1;
        #(100); // Wait for 100ns to count
        if (count !== 10) begin
            $display("Test Case 2 Failed: Counter did not increment correctly. count = %d", count);
            test_passed = 0;
        end

        // Test Case 3: Reset again
        control_signal = 0;
        #(20); // Wait for reset
        if (count !== 0) begin
            $display("Test Case 3 Failed: Counter did not reset. count = %d", count);
            test_passed = 0;
        end

        // Test Case 4: Increment and check specific value
        control_signal = 1;
        #(80); // Enough time to increment the counter
        if (count !== 8) begin
            $display("Test Case 4 Failed: Counter value mismatch. Expected 8, got %d", count);
            test_passed = 0;
        end

        // Complete testing
        #(10); // Small delay before ending tests
        if (test_passed) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error===========");
        end

        // Finish simulation
        $finish;
    end

endmodule
