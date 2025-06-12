`timescale 1ns / 1ps

module testbench;

  reg clk;
  reg [1:0] op_code;
  reg [31:0] operand_a;
  reg [31:0] operand_b;
  wire [31:0] result;
  reg [31:0] expected_result;
  reg test_fail;

  configurable_alu uut (
    .clk(clk),
    .op_code(op_code),
    .operand_a(operand_a),
    .operand_b(operand_b),
    .result(result)
  );

  // Clock generation
  always #10 clk = ~clk;

  // Stimulus generation and response checking
  initial begin
    clk = 0;
    test_fail = 0;

    // Resetting the test inputs
    op_code = 2'b00;
    operand_a = 0;
    operand_b = 0;
    expected_result = 0;
    #100;

    // Test Case 1: Addition
    op_code = 2'b00; // Addition
    operand_a = 32'h00000005;
    operand_b = 32'h00000003;
    expected_result = 32'h00000008;
    #20;
    check_result("Addition");

    // Test Case 2: Subtraction
    op_code = 2'b01; // Subtraction
    operand_a = 32'h00000005;
    operand_b = 32'h00000003;
    expected_result = 32'h00000002;
    #20;
    check_result("Subtraction");

    // Test Case 3: Bitwise AND
    op_code = 2'b10; // AND
    operand_a = 32'h000000FF;
    operand_b = 32'h0000000F;
    expected_result = 32'h0000000F;
    #20;
    check_result("Bitwise AND");

    // Test Case 4: Bitwise OR
    op_code = 2'b11; // OR
    operand_a = 32'h0000FF00;
    operand_b = 32'h00FF0000;
    expected_result = 32'h00FFFF00;
    #20;
    check_result("Bitwise OR");

    if (!test_fail)
      $display("===========Your Design Passed===========");
    else
      $display("===========Error===========");

    $finish;
  end

  // Procedure to check result and display message
  task check_result;
    input [127:0] operation_name;
    begin
      if (result !== expected_result) begin
        $display("%s test failed: Expected %h, Got %h", operation_name, expected_result, result);
        test_fail = 1;
      end else begin
        $display("%s test passed: Expected %h, Got %h", operation_name, expected_result, result);
      end
    end
  endtask

endmodule
