`timescale 1ns / 1ps

module testbench;
    reg [15:0] x, y;
    wire [31:0] prod;
    reg clk, reset;
    reg [31:0] expected_prod;
    reg error_flag;

    // Instantiate the unit under test (UUT)
    multiplier_32bit UUT (
        .x(x),
        .y(y),
        .prod(prod)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Test cases
    initial begin
        // Initialize
        clk = 0;
        reset = 1;
        error_flag = 0;

        // Reset the system
        #10;
        reset = 0;
        #10;
        reset = 1;
        #20;

        // Test case 1: Zero multiplication
        x = 16'd0;
        y = 16'd0;
        expected_prod = 32'd0;
        #10; // Wait for the multiplication to happen
        check_result("Test Case 1");

        // Test case 2: Multiply max values
        x = 16'hFFFF;
        y = 16'h0001;
        expected_prod = 32'h0000FFFF;
        #10; // Wait for the multiplication to happen
        check_result("Test Case 2");

        // Test case 3: Standard multiplication
        x = 16'd25;
        y = 16'd4;
        expected_prod = 32'd100;
        #10; // Wait for the multiplication to happen
        check_result("Test Case 3");

        // Test case 4: Test with different numbers
        x = 16'd1234;
        y = 16'd5678;
        expected_prod = 32'd7006652;
        #10; // Wait for the multiplication to happen
        check_result("Test Case 4");

        // Final results
        if (error_flag) begin
            $display("===========Error===========");
        end else begin
            $display("===========Your Design Passed===========");
        end

        // Finish the simulation
        $finish;
    end

    // Task to check result
    task check_result;
        input [127:0] testname;
        begin
            if (prod != expected_prod) begin
                $display("%s failed: Expected %d, got %d", testname, expected_prod, prod);
                error_flag = 1;
            end else begin
                $display("%s passed", testname);
            end
        end
    endtask

endmodule
