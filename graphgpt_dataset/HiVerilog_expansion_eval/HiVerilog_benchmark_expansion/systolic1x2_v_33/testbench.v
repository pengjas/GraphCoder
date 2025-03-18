`timescale 1ns / 1ps

module tb_dual_mode_counter;

  reg clk;
  reg rst;
  reg mode;
  reg enable;
  wire [7:0] count;

  // Instantiate the Unit Under Test (UUT)
  dual_mode_counter uut (
    .clk(clk),
    .rst(rst),
    .mode(mode),
    .enable(enable),
    .count(count)
  );

  // Clock generation
  initial begin
    clk = 0;
    forever #5 clk = !clk; // Generate a clock with 10ns period
  end

  // Test scenarios
  initial begin
    // Initialize Inputs
    rst = 1;
    mode = 0; 
    enable = 0;
    #10;
    
    rst = 0;  // Release reset
    enable = 1; // Enable counter
    mode = 0; // Set to up-counting mode
    
    // Check up-counting
    #100;
    if (count != 10) begin
      $display("===========Error in Up-Counting Mode===========");
      $finish;
    end

    // Change to down-counting mode
    mode = 1;
    
    // Check down-counting
    #100;
    if (count != 0) begin
      $display("===========Error in Down-Counting Mode===========");
      $finish;
    end
    
    // Check reset functionality
    rst = 1;
    #10;
    if (count != 0) begin
      $display("===========Error in Reset Functionality===========");
      $finish;
    end

    // All tests passed
    $display("===========Your Design Passed===========");
    $finish;
  end

endmodule
