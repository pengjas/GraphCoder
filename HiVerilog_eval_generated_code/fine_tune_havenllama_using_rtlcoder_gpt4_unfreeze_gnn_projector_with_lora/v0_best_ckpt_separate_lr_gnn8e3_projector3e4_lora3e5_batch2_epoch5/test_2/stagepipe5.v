module stagepipe5(
  input clk,
  input rst,
  input [31:0] instr_mem [0:31],
  input [31:0] reg_file [0:31],
  output [31:0] res_reg_file [0:31]
);
wire [31:0] instr;
  wire [31:0] res;
  wire [31:0] reg_file_out;
  wire [31:0] res_reg_file_out;
  wire [31:0] pc;
  wire [31:0] pc_out;
  wire [31:0] alu_out;
  wire [31:0] alu_out_out;

  stage5_fetch fetch_stage(.clk(clk), .rst(rst), .instr_mem(instr_mem), .reg_file(reg_file), .pc(pc), .instr(instr), .pc_out(pc_out), .instr_out(instr_out));
  stage5_decode decode_stage(.clk(clk), .rst(rst), .instr(instr_out), .reg_file(reg_file), .res(res), .pc(pc_out), .res_reg_file_out(res_reg_file_out));
  stage5_execute execute_stage(.clk(clk), .rst(rst), .instr(res), .reg_file(reg_file), .res_reg_file_out(res_reg_file_out), .alu_out(alu_out));
  stage5_memory memory_stage(.clk(clk), .rst(rst), .instr(alu_out), .reg_file(reg_file), .res_reg_file_out(res_reg_file_out), .alu_out(alu_out_out));
  stage5_writeback writeback_stage(.clk(clk), .rst(rst), .instr(alu_out_out), .reg_file(reg_file), .res_reg_file_out(res_reg_file_out), .res(res));

endmodule

module stage5_fetch(
  input clk,
  input rst,
  input [31:0] instr_mem [0:31],
  input [31:0] reg_file [0:31],
  input [31:0] pc,
  output [31:0] instr,
  output [31:0] pc_out,
  output [31:0] instr_out
);
  reg [31:0] instr_reg;
  reg [31:0] pc_reg;
  reg [31:0] instr_out_reg;

  always @(posedge clk, posedge rst) begin
    if (rst) begin
      instr_reg <= 0;
      pc_reg <= 0;
      instr_out_reg <= 0;
    end else begin
      instr_reg <= instr_mem[pc];
      pc_reg <= pc + 1;
      instr_out_reg <= instr_reg;
    end
  end

  assign instr = instr_reg;
  assign pc_out = pc_reg;
  assign instr_out = instr_out_reg;
endmodule

module stage5_decode(
  input clk,
  input rst,
  input [31:0] instr,
  input [31:0] reg_file [0:31],
  output [31:0] res,
  output [31:0] pc,
  output [31:0] res_reg_file_out
);
  reg [31:0] res_reg;
  reg [31:0] pc_reg;
  reg [31:0] res_reg_file_out_reg;
  reg [31:0] reg_file_out_reg;
  reg [31:0] rs1_reg;
  reg [31:0] rs2_reg;
  reg [31:0] rd_reg;
  reg [31:0] op_reg;

  always @(posedge clk, posedge rst) begin
    if (rst) begin
      res_reg <= 0;
      pc_reg <= 0;
      res_reg_file_out_reg <= 0;
      reg_file_out_reg <= 0;
endmodule