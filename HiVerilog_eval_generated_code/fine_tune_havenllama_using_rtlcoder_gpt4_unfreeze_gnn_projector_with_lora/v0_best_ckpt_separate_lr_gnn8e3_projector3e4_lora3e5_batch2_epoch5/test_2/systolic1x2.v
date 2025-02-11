module systolic1x2 (
  input clk,
  input rst,
  input [15:0] a0,
  input [15:0] a1,
  input [15:0] b0,
  output [31:0] c0,
  output [31:0] c1
);

  reg [15:0] b1;

  always @(posedge clk) begin
    if (rst) begin
      b1 <= 0;
    end else begin
      b1 <= b0;
    end
  end

  assign c0 = a0 * b0;
  assign c1 = a1 * b1;

endmodule