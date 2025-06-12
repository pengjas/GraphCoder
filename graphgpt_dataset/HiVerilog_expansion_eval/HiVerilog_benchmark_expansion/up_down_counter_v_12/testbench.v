`timescale 1ns / 1ps

module tb_multi_function_alu;
  
  reg [2:0] op_code;
  reg [31:0] operand_a, operand_b;
  wire [31:0] result;
  wire zero;
  
  // Instantiate the multi_function_alu
  multi_function_alu UUT (
    .op_code(op_code),
    .operand_a(operand_a),
    .operand_b(operand_b),
    .result(result),
    .zero(zero)
  );
  
  // Clock generation
  reg clk;
  initial clk = 0;
  always #5 clk = ~clk;
  
  // Reset and Test Case Control
  reg reset;
  integer i;
  reg [31:0] expected_result;
  reg expected_zero;
  reg pass;
  
  initial begin
    // Initialize inputs
    op_code = 3'b000;
    operand_a = 0;
    operand_b = 0;
    reset = 1;
    pass = 1;
    
    // Apply Reset
    #10;
    reset = 0;

    // Test case 1: ADD operation
    op_code = 3'b000; // ADD
    operand_a = 32'h0000_0001;
    operand_b = 32'h0000_0001;
    expected_result = 32'h0000_0002;
    expected_zero = 0;
    #10;  // Wait for operation
    check_result("ADD");
    
    // Test case 2: SUB operation
    op_code = 3'b001; // SUB
    operand_a = 32'h0000_0003;
    operand_b = 32'h0000_0001;
    expected_result = 32'h0000_0002;
    expected_zero = 0;
    #10;  // Wait for operation
    check_result("SUB");
    
    // Test case 3: AND operation
    op_code = 3'b010; // AND
    operand_a = 32'h0000_0003;
    operand_b = 32'h0000_0001;
    expected_result = 32'h0000_0001;
    expected_zero = 0;
    #10;  // Wait for operation
    check_result("AND");
    
    // Test case 4: OR operation
    op_code = 3'b011; // OR
    operand_a = 32'h0000_0002;
    operand_b = 32'h0000_0001;
    expected_result = 32'h0000_0003;
    expected_zero = 0;
    #10;  // Wait for operation
    check_result("OR");
    
    // Test case 5: XOR operation
    op_code = 3'b100; // XOR
    operand_a = 32'h0000_0003;
    operand_b = 32'h0000_0001;
    expected_result = 32'h0000_0002;
    expected_zero = 0;
    #10;  // Wait for operation
    check_result("XOR");
    
    // Final pass/fail message
    if (pass) begin
      $display("===========Your Design Passed===========");
    end
    else begin
      $display("===========Error===========");
    end
    
    // Finish simulation
    $finish;
  end

  task check_result;
    input [8*8:1] testname;
    begin
      if (result !== expected_result || zero !== expected_zero) begin
        $display("Test %s FAILED: Expected result = %h, got %h, Expected zero = %b, got %b",
                 testname, expected_result, result, expected_zero, zero);
        pass = 0;
      end
      else begin
        $display("Test %s PASSED", testname);
      end
    end
  endtask
  
endmodule
