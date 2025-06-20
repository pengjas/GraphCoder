`timescale 1ns / 1ps

module tb_multiplier_4bit;

    reg [3:0] a, b;
    wire [7:0] product;
    reg clk, reset;
    integer i, j, error_count;

    // Instantiate the Unit Under Test (UUT)
    multiplier_4bit uut (
        .a(a),
        .b(b),
        .product(product)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Reset generation
    initial begin
        clk = 0;
        reset = 1;
        #10;
        reset = 0;
    end

    // Test cases
    initial begin
        error_count = 0;
        // Apply inputs
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                a = i;
                b = j;
                #10; // wait for the product to be calculated

                // Check the result
                if (product !== a * b) begin
                    $display("Error: a = %d, b = %d, Expected product = %d, Got = %d",
                             a, b, a * b, product);
                    error_count = error_count + 1;
                end
            end
        end

        // Check if there were any errors during the test
        if (error_count == 0) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error: %d mismatches found===========", error_count);
        end

        $finish;
    end

endmodule
