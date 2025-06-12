`timescale 1ns / 1ps

module testbench;

    reg [31:0] X, Y;
    wire [31:0] Sum;
    wire Carry_out;
    reg clk, reset;
    reg [31:0] expected_sum;
    reg expected_carry;
    integer test_cases = 0, passed_cases = 0;

    // Instantiate the adder_32bit module
    adder_32bit UUT (
        .X(X),
        .Y(Y),
        .Sum(Sum),
        .Carry_out(Carry_out)
    );

    // Clock generation
    always begin
        clk = 1; #5;
        clk = 0; #5;
    end

    // Initial block for tests
    initial begin
        // Reset
        reset = 1; #10;
        reset = 0;

        // Test case 1
        X = 32'h00000001; Y = 32'h00000001;
        expected_sum = 32'h00000002; expected_carry = 0;
        #10; // Wait for the operation
        check_result(expected_sum, expected_carry);

        // Test case 2
        X = 32'hFFFFFFFF; Y = 32'h00000001;
        expected_sum = 32'h00000000; expected_carry = 1;
        #10; // Wait for the operation
        check_result(expected_sum, expected_carry);

        // Test case 3
        X = 32'h80000000; Y = 32'h80000000;
        expected_sum = 32'h00000000; expected_carry = 1;
        #10; // Wait for the operation
        check_result(expected_sum, expected_carry);

        // Test case 4
        X = 32'h12345678; Y = 32'h87654321;
        expected_sum = 32'h99999999; expected_carry = 0;
        #10; // Wait for the operation
        check_result(expected_sum, expected_carry);

        // Final result check
        if (passed_cases == test_cases) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error===========");
        end

        $finish;
    end

    // Task to check result
    task check_result;
        input [31:0] expected_sum;
        input expected_carry;
        begin
            test_cases = test_cases + 1;
            if (Sum === expected_sum && Carry_out === expected_carry) begin
                $display("Test Case %d Passed: Sum = %h, Carry_out = %b", test_cases, Sum, Carry_out);
                passed_cases = passed_cases + 1;
            end else begin
                $display("Test Case %d Failed: Expected Sum = %h, Output Sum = %h, Expected Carry = %b, Output Carry = %b",
                         test_cases, expected_sum, Sum, expected_carry, Carry_out);
            end
        end
    endtask

endmodule
