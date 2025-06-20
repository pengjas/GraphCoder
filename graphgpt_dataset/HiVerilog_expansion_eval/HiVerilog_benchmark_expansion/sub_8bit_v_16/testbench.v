`timescale 1ns / 1ps

module tb_mul_4bit;

    // Inputs
    reg [3:0] X;
    reg [3:0] Y;

    // Outputs
    wire [7:0] P;

    // Instantiate the Unit Under Test (UUT)
    mul_4bit uut (
        .X(X), 
        .Y(Y), 
        .P(P)
    );

    // Clock Generation
    reg clk = 0;
    always #10 clk = ~clk;  // Clock with period 20ns

    // Reset Generation
    reg rst_n;
    initial begin
        rst_n = 0; // Active low reset
        #15;
        rst_n = 1; // Release reset after 15ns
    end

    // Test Cases
    integer errors = 0;
    initial begin
        // Initialize Inputs
        X = 0;
        Y = 0;

        // Wait for reset release
        @(posedge rst_n);
        @(posedge clk);  // Wait for one clock cycle after reset release

        // Test Case 1
        X = 4'b0011;  // 3
        Y = 4'b0101;  // 5
        #20;  // Wait for 20ns for multiplication to process (two clock cycles)
        if (P !== 8'd15) begin
            $display("Test Case 1 Failed: X=%b, Y=%b, Expected P=%d, Received P=%d", X, Y, 8'd15, P);
            errors = errors + 1;
        end

        // Test Case 2
        X = 4'b0100;  // 4
        Y = 4'b0110;  // 6
        #20;
        if (P !== 8'd24) begin
            $display("Test Case 2 Failed: X=%b, Y=%b, Expected P=%d, Received P=%d", X, Y, 8'd24, P);
            errors = errors + 1;
        end
        
        // Test Case 3
        X = 4'b1111;  // 15
        Y = 4'b1111;  // 15
        #20;
        if (P !== 8'd225) begin
            $display("Test Case 3 Failed: X=%b, Y=%b, Expected P=%d, Received P=%d", X, Y, 8'd225, P);
            errors = errors + 1;
        end
        
        if (errors == 0) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error=========== Number of Errors: %d", errors);
        end

        $finish;
    end
endmodule
