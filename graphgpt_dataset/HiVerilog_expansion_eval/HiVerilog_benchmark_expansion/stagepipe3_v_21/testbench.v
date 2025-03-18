`timescale 1ns / 1ps

module testbench;

    // Inputs
    reg clk;
    reg rst;
    reg [1:0] opcode;
    reg [7:0] data_a;
    reg [7:0] data_b;

    // Outputs
    wire [15:0] result;
    wire valid;

    // Instantiate the Unit Under Test (UUT)
    arithmetic_unit uut (
        .clk(clk),
        .rst(rst),
        .opcode(opcode),
        .data_a(data_a),
        .data_b(data_b),
        .result(result),
        .valid(valid)
    );

    // Clock generation
    always #5 clk = ~clk;

    // Test variables
    reg [15:0] expected_result;
    reg expected_valid;
    integer errors = 0;

    initial begin
        // Initialize Inputs
        clk = 0;
        rst = 1;
        opcode = 0;
        data_a = 0;
        data_b = 0;

        // Wait for global reset
        #100;
        rst = 0;

        // Test Case 1: Add Operation
        #10;
        opcode = 2'b00;  // Add
        data_a = 8'd10;
        data_b = 8'd25;
        expected_result = 10 + 25;
        expected_valid = 1;
        #10;
        check_results("Add Operation");

        // Test Case 2: Subtract Operation
        #10;
        opcode = 2'b01;  // Subtract
        data_a = 8'd50;
        data_b = 8'd20;
        expected_result = 50 - 20;
        expected_valid = 1;
        #10;
        check_results("Subtract Operation");

        // Test Case 3: Multiply Operation
        #10;
        opcode = 2'b10;  // Multiply
        data_a = 8'd5;
        data_b = 8'd6;
        expected_result = 5 * 6;
        expected_valid = 1;
        #10;
        check_results("Multiply Operation");

        // Test Case 4: Divide Operation
        #10;
        opcode = 2'b11;  // Divide
        data_a = 8'd40;
        data_b = 8'd5;
        expected_result = 40 / 5;
        expected_valid = 1;
        #10;
        check_results("Divide Operation");

        // Test Case 5: Divide by Zero
        #10;
        opcode = 2'b11;  // Divide
        data_a = 8'd40;
        data_b = 8'd0;
        expected_result = 0;  // undefined result
        expected_valid = 0;   // invalid output
        #10;
        check_results("Divide by Zero");

        // Final report
        if (errors == 0) begin
            $display("===========Your Design Passed===========");
        end else begin
            $display("===========Error: %d failures===========", errors);
        end

        $finish;
    end

    task check_results;
        input [100*8:1] test_name;
        begin
            if (result !== expected_result || valid !== expected_valid) begin
                $display("Error in %s: Expected result = %d, got %d; Expected valid = %b, got %b",
                         test_name, expected_result, result, expected_valid, valid);
                errors = errors + 1;
            end else begin
                $display("%s Passed", test_name);
            end
        end
    endtask

endmodule
