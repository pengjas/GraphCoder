`timescale 1ns / 1ps

module tb_subtractor_8bit;
    reg [7:0] a, b;
    reg bin;
    wire [7:0] diff;
    wire bout;

    // Instantiate the Unit Under Test (UUT)
    subtractor_8bit uut(
        .a(a),
        .b(b),
        .bin(bin),
        .diff(diff),
        .bout(bout)
    );

    // Clock and Reset Signals
    reg clk;
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // Toggle clock every 5 ns
    end

    // Test cases and result checking
    integer num_errors;
    
    initial begin
        // Initialize Inputs
        a = 0;
        b = 0;
        bin = 0;
        num_errors = 0;

        // Apply Reset
        #100;
        
        // Test Case 1: Simple subtraction without borrow
        a = 8'd15;  // 15
        b = 8'd7;   // 7
        bin = 1'b0;
        #10; // Wait for propagation
        if (diff !== 8'd8 || bout !== 1'b0) begin
            $display("Error: Test Case 1 Failed. Expected diff=8, bout=0, got diff=%d, bout=%d", diff, bout);
            num_errors = num_errors + 1;
        end

        // Test Case 2: Subtraction with borrow in
        a = 8'd10;  // 10
        b = 8'd15;  // 15
        bin = 1'b0;
        #10; // Wait for propagation
        if (diff !== 8'd251 || bout !== 1'b1) begin // 10 - 15 - 0 = 251 (Two's Complement) and borrow
            $display("Error: Test Case 2 Failed. Expected diff=251, bout=1, got diff=%d, bout=%d", diff, bout);
            num_errors = num_errors + 1;
        end
        
        // Test Case 3: Borrow-in and larger B
        a = 8'd5;  // 5
        b = 8'd8;  // 8
        bin = 1'b1;
        #10; // Wait for propagation
        if (diff !== 8'd252 || bout !== 1'b1) begin // 5 - 8 - 1 = 252 (Two's Complement) and borrow
            $display("Error: Test Case 3 Failed. Expected diff=252, bout=1, got diff=%d, bout=%d", diff, bout);
            num_errors = num_errors + 1;
        end

        // Conclusion
        if (num_errors === 0) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error=========== Total Errors: %d", num_errors);
        end
        
        // Finish simulation
        $finish;
    end

endmodule
