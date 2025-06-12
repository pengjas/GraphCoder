`timescale 1ns / 1ps

module tb_matrix_multiplier2x2;

  reg clk;
  reg rst;
  reg [7:0] x0, x1, y0, y1;
  wire [15:0] p0, p1, p2, p3;

  matrix_multiplier2x2 uut (
    .clk(clk),
    .rst(rst),
    .x0(x0),
    .x1(x1),
    .y0(y0),
    .y1(y1),
    .p0(p0),
    .p1(p1),
    .p2(p2),
    .p3(p3)
  );

  // Clock generation
  always begin
    #5 clk = ~clk; // 100MHz clock
  end

  // Test cases and result checking
  initial begin
    $monitor("At time %t, p0 = %d, p1 = %d, p2 = %d, p3 = %d",
              $time, p0, p1, p2, p3);

    // Initialize inputs
    clk = 0;
    rst = 1; // Assert reset
    x0 = 0; x1 = 0; y0 = 0; y1 = 0;

    // Release reset
    #10 rst = 0;

    // Test case 1
    x0 = 8; x1 = 2; y0 = 4; y1 = 3;
    #10;
    if (p0 !== 8*4 || p1 !== 2*4 || p2 !== 8*3 || p3 !== 2*3) begin
      $display("===========Error in Test Case 1===========");
      $finish;
    end

    // Test case 2
    x0 = 10; x1 = 5; y0 = 2; y1 = 6;
    #10;
    if (p0 !== 10*2 || p1 !== 5*2 || p2 !== 10*6 || p3 !== 5*6) begin
      $display("===========Error in Test Case 2===========");
      $finish;
    end

    // Test case 3
    x0 = 255; x1 = 255; y0 = 1; y1 = 1;
    #10;
    if (p0 !== 255*1 || p1 !== 255*1 || p2 !== 255*1 || p3 !== 255*1) begin
      $display("===========Error in Test Case 3===========");
      $finish;
    end

    $display("===========Your Design Passed===========");
    $finish;
  end

endmodule
