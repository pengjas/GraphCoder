`timescale 1ns / 1ps

module tb_modular_alu;

  reg [1:0] op_code;
  reg [31:0] a, b;
  wire [31:0] result;
  wire zero;
  reg clk, reset;
  
  // Instantiate the modular_alu
  modular_alu UUT (
    .op_code(op_code),
    .a(a),
    .b(b),
    .result(result),
    .zero(zero)
  );

  // Clock generation
  always #5 clk = ~clk;

  // Test cases
  initial begin
    clk = 0;
    reset = 1;
    #10 reset = 0;
    
    // Test 1: Addition a + b
    op_code = 2'b00;
    a = 32'd10;
    b = 32'd20;
    #10;
    if (result != 32'd30 || zero != 1'b0) begin
      $display("===========Error in Addition===========");
      $finish;
    end
    
    // Test 2: Subtraction a - b
    op_code = 2'b01;
    a = 32'd50;
    b = 32'd20;
    #10;
    if (result != 32'd30 || zero != 1'b0) begin
      $display("===========Error in Subtraction===========");
      $finish;
    end
    
    // Test 3: Bitwise AND a & b
    op_code = 2'b10;
    a = 32'hFF00FF00;
    b = 32'h0FF00FF0;
    #10;
    if (result != 32'h0F000F00 || zero != 1'b0) begin
      $display("===========Error in AND operation===========");
      $finish;
    end
    
    // Test 4: Bitwise OR a | b
    op_code = 2'b11;
    a = 32'h12345678;
    b = 32'h87654321;
    #10;
    if (result != 32'h97755779 || zero != 1'b0) begin
      $display("===========Error in OR operation===========");
      $finish;
    end

    // Test 5: Zero detection
    op_code = 2'b00;
    a = 32'd0;
    b = 32'd0;
    #10;
    if (result != 32'd0 || zero != 1'b1) begin
      $display("===========Error in Zero detection===========");
      $finish;
    end

    $display("===========Your Design Passed===========");
    $finish;
  end

endmodule
