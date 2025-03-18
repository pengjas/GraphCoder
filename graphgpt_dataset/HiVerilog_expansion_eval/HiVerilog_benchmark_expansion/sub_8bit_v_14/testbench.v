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

  // Clock and reset variables
  reg clk;
  reg rst;

  // Clock generation
  always #5 clk = ~clk; // Generate a clock with period 10 ns

  // Reset generation
  initial begin
    rst = 1; // Assert reset
    #15;
    rst = 0; // Deassert reset
  end

  // Test cases
  integer errors = 0;
  initial begin
    // Initialize Inputs
    clk = 0;
    X = 0;
    Y = 0;

    // Wait for reset deassertion
    wait(rst == 0);
    #10;
    
    // Test Case 1
    X = 8'd15;  // 15
    Y = 8'd10;  // 10
    #10;        // Wait for one clock cycle
    if (P !== 16'd150) begin
      $display("Error: X = %d, Y = %d, Expected P = %d, Got P = %d", X, Y, 150, P);
      errors = errors + 1;
    end

    // Test Case 2
    X = 8'd25;  // 25
    Y = 8'd4;   // 4
    #10;
    if (P !== 16'd100) begin
      $display("Error: X = %d, Y = %d, Expected P = %d, Got P = %d", X, Y, 100, P);
      errors = errors + 1;
    end

    // Test Case 3
    X = 8'd0;   // 0
    Y = 8'd255; // 255
    #10;
    if (P !== 16'd0) begin
      $display("Error: X = %d, Y = %d, Expected P = %d, Got P = %d", X, Y, 0, P);
      errors = errors + 1;
    end

    // Test Case 4
    X = 8'd127; // 127
    Y = 8'd2;   // 2
    #10;
    if (P !== 16'd254) begin
      $display("Error: X = %d, Y = %d, Expected P = %d, Got P = %d", X, Y, 254, P);
      errors = errors + 1;
    end

    // Test Results
    if (errors == 0) begin
      $display("===========Your Design Passed===========");
    end else begin
      $display("===========Error===========: %d Errors detected", errors);
    end

    $finish;
  end

endmodule
