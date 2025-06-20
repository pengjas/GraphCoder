`timescale 1ns / 1ps

module tb_multiplier;

  reg [31:0] a;
  reg [31:0] b;
  reg multc;
  reg clk, reset;
  wire [63:0] y;
  wire [31:0] msb;
  wire [31:0] lsb;
  wire zero;

  // Instantiate the Unit Under Test (UUT)
  multiplier uut (
    .a(a),
    .b(b),
    .multc(multc),
    .y(y),
    .msb(msb),
    .lsb(lsb),
    .zero(zero)
  );

  // Clock generation
  always #10 clk = ~clk;

  // Initial block
  initial begin
    // Initialize Inputs
    clk = 0;
    reset = 1;
    a = 0;
    b = 0;
    multc = 0;

    // Reset pulse
    #100;
    reset = 0;
    #100;
    reset = 1;
    #100;

    // Test case 1: Small signed multiplication
    a = 32'd15;
    b = 32'd3;
    multc = 1'b0;  // Signed multiplication
    #20;
    verify_results(45, 1'b0);

    // Test case 2: Small unsigned multiplication
    a = 32'd15;
    b = 32'd3;
    multc = 1'b1;  // Unsigned multiplication
    #20;
    verify_results(45, 1'b0);

    // Test case 3: Negative signed multiplication
    a = -32'd15;
    b = 32'd3;
    multc = 1'b0;
    #20;
    verify_results(-45, 1'b0);

    // Test case 4: Larger signed multiplication
    a = 32'h7FFF_FFFF;  // Largest positive 32-bit integer
    b = 32'd2;
    multc = 1'b0;
    #20;
    verify_results(64'h0000_0000_FFFF_FFFE, 1'b0);

    // Test case 5: Result is zero
    a = 0;
    b = 32'h12345678;
    multc = 1'b0;
    #20;
    verify_results(0, 1'b1);

    // Finish test
    $display("===========Your Design Passed===========");
    $finish;
  end

  task verify_results;
    input [63:0] expected_y;
    input expected_zero;
    begin
      if (y !== expected_y || zero !== expected_zero) begin
        $display("===========Error===========");
        $display("Error at a = %d, b = %d, multc = %b", a, b, multc);
        $display("Expected y = %h, Got y = %h", expected_y, y);
        $display("Expected zero = %b, Got zero = %b", expected_zero, zero);
        $finish;
      end
    end
  endtask

endmodule
