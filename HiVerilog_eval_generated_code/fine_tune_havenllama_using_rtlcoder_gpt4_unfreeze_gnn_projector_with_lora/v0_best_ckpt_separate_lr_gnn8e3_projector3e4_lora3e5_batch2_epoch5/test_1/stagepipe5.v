module stagepipe5(
  input clk,
  input rst,
  input [31:0] instr_mem [0:31],
  input [31:0] reg_file [0:31],
  output [31:0] res_reg_file [0:31]
);
wire [31:0] instr_mem_out;
  wire [31:0] reg_file_out;
  wire [31:0] alu_out;
  wire [31:0] res_reg_file_out;
  wire [31:0] pc_out;

  stagepipe5_fetch fetch_stage(
    .clk(clk),
    .rst(rst),
    .instr_mem(instr_mem),
    .reg_file(reg_file),
    .res_reg_file(res_reg_file),
    .pc_out(pc_out)
  );

  stagepipe5_decode decode_stage(
    .clk(clk),
    .rst(rst),
    .instr_mem_out(instr_mem_out),
    .reg_file_out(reg_file_out),
    .alu_out(alu_out)
  );

  stagepipe5_execute execute_stage(
    .clk(clk),
    .rst(rst),
    .alu_out(alu_out),
    .res_reg_file_out(res_reg_file_out)
  );

  stagepipe5_memory memory_stage(
    .clk(clk),
    .rst(rst),
    .alu_out(alu_out),
    .res_reg_file_out(res_reg_file_out)
  );

  stagepipe5_writeback writeback_stage(
    .clk(clk),
    .rst(rst),
    .alu_out(alu_out),
    .res_reg_file_out(res_reg_file_out),
    .reg_file(reg_file)
  );

  assign res_reg_file[0] = res_reg_file_out[0];
  assign res_reg_file[1] = res_reg_file_out[1];
  assign res_reg_file[2] = res_reg_file_out[2];
  assign res_reg_file[3] = res_reg_file_out[3];
  assign res_reg_file[4] = res_reg_file_out[4];
  assign res_reg_file[5] = res_reg_file_out[5];
  assign res_reg_file[6] = res_reg_file_out[6];
  assign res_reg_file[7] = res_reg_file_out[7];
  assign res_reg_file[8] = res_reg_file_out[8];
  assign res_reg_file[9] = res_reg_file_out[9];
  assign res_reg_file[10] = res_reg_file_out[10];
  assign res_reg_file[11] = res_reg_file_out[11];
  assign res_reg_file[12] = res_reg_file_out[12];
  assign res_reg_file[13] = res_reg_file_out[13];
  assign res_reg_file[14] = res_reg_file_out[14];
  assign res_reg_file[15] = res_reg_file_out[15];
  assign res_reg_file[16] = res_reg_file_out[16];
  assign res_reg_file[17] = res_reg_file_out[17];
  assign res_reg_file[18] = res_reg_file_out[18];
  assign res_reg_file[19] = res_reg_file_out[19];
  assign res_reg_file[20] = res_reg_file_out[20];
  assign res_reg_file[21] = res_reg_file_out[21];
  assign res_reg_file[22] = res_reg_file_out[22];
  assign res_reg_file[23] = res_reg_file_out[23];
  assign res_reg_file[24] = res_reg_file_out[24];
  assign res_reg_file[25] = res_reg
endmodule