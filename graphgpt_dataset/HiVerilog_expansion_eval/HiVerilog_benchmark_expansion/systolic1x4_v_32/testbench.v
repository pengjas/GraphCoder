`timescale 1ns / 1ps

module tb_cyclic_rotator4;

  // Inputs
  reg en;
  reg rot_dir;
  reg [3:0] in_data;

  // Outputs
  wire [3:0] out_data;

  // Instantiate the Unit Under Test (UUT)
  cyclic_rotator4 uut (
    .en(en),
    .rot_dir(rot_dir),
    .in_data(in_data),
    .out_data(out_data)
  );

  // Clock generation
  reg clk;
  initial begin
    clk = 0;
    forever #5 clk = !clk;
  end

  // Reset and enable generation
  initial begin
    // Initialize Inputs
    en = 0;
    rot_dir = 0;
    in_data = 4'b0000;

    // Wait for global reset to finish
    @(posedge clk);
    en = 1; // Enable the module
    rot_dir = 0; // Set rotation direction to right
    in_data = 4'b1001;

    @(posedge clk);
    if (out_data !== 4'b1100) $display("===========Error===========");
    
    rot_dir = 1; // Change rotation direction to left
    in_data = 4'b1001;

    @(posedge clk);
    if (out_data !== 4'b0011) $display("===========Error===========");

    in_data = 4'b0001; // Test edge case
    @(posedge clk);
    if (out_data !== 4'b0010 && rot_dir == 1) $display("===========Error===========");
    
    in_data = 4'b0001; // Test edge case
    rot_dir = 0; // Rotate right
    @(posedge clk);
    if (out_data !== 4'b1000) $display("===========Error===========");

    // Disable and check for no rotation
    en = 0;
    in_data = 4'b0110;
    @(posedge clk);
    if (out_data !== 4'b0110) $display("===========Error===========");
    
    // Add more checks if necessary
    $display("===========Your Design Passed===========");
    $finish;
  end
      
endmodule
