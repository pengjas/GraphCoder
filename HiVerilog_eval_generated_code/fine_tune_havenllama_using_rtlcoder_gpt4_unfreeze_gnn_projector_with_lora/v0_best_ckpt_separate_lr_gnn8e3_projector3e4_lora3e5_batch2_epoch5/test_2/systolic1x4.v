module systolic1x4(
  input clk,
  input rst,
  input [15:0] a0,
  input [15:0] a1,
  input [15:0] a2,
  input [15:0] a3,
  input [15:0] b0,
  output [31:0] c0,
  output [31:0] c1,
  output [31:0] c2,
  output [31:0] c3
);

  reg [15:0] b1, b2, b3;
  reg [31:0] c0_reg, c1_reg, c2_reg, c3_reg;

  always @(posedge clk) begin
    if (rst) begin
      b1 <= 16'b0;
      b2 <= 16'b0;
      b3 <= 16'b0;
      c0_reg <= 32'b0;
      c1_reg <= 32'b0;
      c2_reg <= 32'b0;
      c3_reg <= 32'b0;
    end else begin
      b1 <= b0;
      b2 <= b1;
      b3 <= b2;
      c0_reg <= c0_reg + (a0 * b0);
      c1_reg <= c1_reg + (a1 * b1);
      c2_reg <= c2_reg + (a2 * b2);
      c3_reg <= c3_reg + (a3 * b3);
    end
  end

  assign c0 = c0_reg;
  assign c1 = c1_reg;
  assign c2 = c2_reg;
  assign c3 = c3_reg;

endmodule