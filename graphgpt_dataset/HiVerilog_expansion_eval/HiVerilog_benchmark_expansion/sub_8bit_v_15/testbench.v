`timescale 1ns / 1ps

module tb_mult_8bit;

    // Inputs
    reg [7:0] X;
    reg [7:0] Y;

    // Outputs
    wire [15:0] P;

    // Instantiate the Unit Under Test (UUT)
    mult_8bit uut (
        .X(X), 
        .Y(Y), 
        .P(P)
    );

    // Clock generation
    reg clk;
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // Reset generation
    reg reset;
    initial begin
        reset = 1;
        #10;
        reset = 0;
    end

    // Test cases
    integer i;
    initial begin
        // Initialize Inputs
        X = 0;
        Y = 0;

        // Wait for reset release
        wait (reset === 0);
        #10;

        // Test case 1: Simple multiplication
        X = 8'd12; Y = 8'd10;
        #10;  // Wait a clock cycle for output
        check_result(X * Y, P);

        // Test case 2: Zero multiplication
        X = 8'd0; Y = 8'd57;
        #10;  // Wait a clock cycle for output
        check_result(X * Y, P);

        // Test case 3: Maximum value multiplication
        X = 8'hFF; Y = 8'hFF;
        #10;  // Wait a clock cycle for output
        check_result(X * Y, P);

        // Test case 4: Random cases
        for (i = 0; i < 10; i = i + 1) begin
            X = $random % 256;
            Y = $random % 256;
            #10;  // Wait a clock cycle for output
            check_result(X * Y, P);
        end

        // All tests passed
        $display("===========Your Design Passed===========");
        $finish;
    end

    // Task for checking result
    task check_result;
        input [15:0] expected;
        input [15:0] actual;
        begin
            if (expected !== actual) begin
                $display("===========Error: Expected %d, got %d===========", expected, actual);
                $finish;
            end
        end
    endtask

endmodule
