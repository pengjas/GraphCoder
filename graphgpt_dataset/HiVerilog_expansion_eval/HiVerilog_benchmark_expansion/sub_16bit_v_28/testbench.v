`timescale 1ns / 1ps

module tb_parity_16bit;

    reg [7:0] X, Y;
    wire P;
    reg clk, rst;
    reg [15:0] combined_input;
    reg expected_parity;
    integer errors, i;

    // Instantiating the module under test
    parity_16bit UUT (
        .X(X),
        .Y(Y),
        .P(P)
    );

    // Clock generation
    always #5 clk = ~clk; // Clock with period of 10 ns

    // Reset generation
    initial begin
        rst = 1;
        #10;
        rst = 0;
    end

    // Initialize variables
    initial begin
        clk = 0;
        rst = 1;
        X = 0;
        Y = 0;
        errors = 0;
        #15;
        rst = 0;

        // Test cases
        for (i = 0; i < 256; i = i + 1) begin
            X = i; 
            Y = 255 - i; // Complementary values for more edge case testing
            #10; // Wait for clock edge

            // Calculate the expected even parity
            combined_input = {X, Y};
            expected_parity = ^combined_input;

            // Check result
            if (P !== expected_parity) begin
                $display("Error: Input X = %b, Y = %b, Expected Parity = %b, Output Parity = %b", X, Y, expected_parity, P);
                errors = errors + 1;
            end
        end

        // Finish simulation with pass/fail message
        if (errors == 0) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error===========");
        end

        $finish;
    end

endmodule
