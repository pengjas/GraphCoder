module dual_mode_counter (
  input clk,
  input rst,
  input mode,
  input enable,
  output [7:0] count
);
  wire [7:0] next_count;
  Counter counter_inst (
    .clk(clk),
    .rst(rst),
    .mode(mode),
    .enable(enable),
    .count(next_count)
  );
  assign count = next_count;
endmodule

module Counter (
  input clk,
  input rst,
  input mode,
  input enable,
  output reg [7:0] count
);
  always @(posedge clk or posedge rst) begin
    if (rst) begin
      count <= 0;
    end
    else if (enable) begin
      if (mode == 0)  // 0 for up-counting
        count <= count + 1;
      else            // 1 for down-counting
        count <= count - 1;
    end
  end
endmodule
