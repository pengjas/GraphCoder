`timescale 1ns / 1ps

module test_multiplier_4bit;

    // Inputs
    reg [3:0] a;
    reg [3:0] b;

    // Outputs
    wire [7:0] product;

    // Instantiate the Unit Under Test (UUT)
    multiplier_4bit uut (
        .a(a), 
        .b(b), 
        .product(product)
    );

    // Test variables
    integer i, j, passed;
    reg [7:0] expected_product;

    // Clock and reset generation
    reg clk;
    initial begin
        clk = 0;
        forever #5 clk = !clk;
    end

    initial begin
        // Initialize Inputs
        a = 0;
        b = 0;
        passed = 1;

        // Wait for Global Reset to finish
        #100;

        // Add stimulus here
        for (i = 0; i < 16; i = i + 1) begin
            for (j = 0; j < 16; j = j + 1) begin
                a = i;
                b = j;
                expected_product = i * j; // Expected result of multiplication
                #10; // Wait for combinational logic to settle

                if (product !== expected_product) begin
                    $display("Error: a=%d, b=%d, expected_product=%d, got_product=%d", a, b, expected_product, product);
                    passed = 0;
                end
            end
        end

        // Check if all cases passed
        if (passed) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error===========");
        end

        // Finish the simulation
        $finish;
    end

endmodule
