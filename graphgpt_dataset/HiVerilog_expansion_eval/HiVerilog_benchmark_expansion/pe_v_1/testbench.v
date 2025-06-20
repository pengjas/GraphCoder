`timescale 1ns / 1ps

module tb_ds;

  reg clk;
  reg rst;
  reg [31:0] x;
  reg [31:0] y;
  reg [31:0] z;
  wire [31:0] quotient;
  wire [31:0] difference;
  
  integer errors;

  // Instantiate the Unit Under Test (UUT)
  ds uut (
    .clk(clk),
    .rst(rst),
    .x(x),
    .y(y),
    .z(z),
    .quotient(quotient),
    .difference(difference)
  );

  // Clock generation
  always #5 clk = ~clk;

  // Test cases
  initial begin
    // Initialize Inputs
    clk = 0;
    rst = 1;
    x = 0;
    y = 0;
    z = 0;
    errors = 0;
    
    // Reset the system
    #10;
    rst = 0;
    #20;
    
    // Apply Inputs
    // Test Case 1: x = 100, y = 4, z = 10
    x = 32'd100;
    y = 32'd4;
    z = 32'd10;
    #10;  // Wait for the operation to complete
    
    if (quotient != 25 || difference != 15) begin
      $display("Error with test case 1: x = 100, y = 4, z = 10");
      errors = errors + 1;
    end
    
    // Test Case 2: x = 500, y = 20, z = 20
    x = 32'd500;
    y = 32'd20;
    z = 32'd20;
    #10;  // Wait for the operation to complete
    
    if (quotient != 25 || difference != 5) begin
      $display("Error with test case 2: x = 500, y = 20, z = 20");
      errors = errors + 1;
    end
    
    // Test Case 3: x = 1024, y = 32, z = 32
    x = 32'd1024;
    y = 32'd32;
    z = 32'd32;
    #10;  // Wait for the operation to complete
    
    if (quotient != 32 || difference != 0) begin
      $display("Error with test case 3: x = 1024, y = 32, z = 32");
      errors = errors + 1;
    end
    
    // Check all test cases and finalize the test
    if (errors == 0) begin
      $display("===========Your Design Passed===========");
    end else begin
      $display("===========Error: %d failures===========", errors);
    end

    $finish;
  end

endmodule
