module cyclic_rotator4(
  input en,
  input rot_dir,
  input [3:0] in_data,
  output [3:0] out_data
);

wire [3:0] left_rotated_data;
wire [3:0] right_rotated_data;

left_rotate lr(
    .data(in_data),
    .rotated_data(left_rotated_data)
);

right_rotate rr(
    .data(in_data),
    .rotated_data(right_rotated_data)
);

assign out_data = (en) ? ((rot_dir) ? left_rotated_data : right_rotated_data) : in_data;

endmodule

module left_rotate(
  input [3:0] data,
  output [3:0] rotated_data
);
  assign rotated_data = {data[2:0], data[3]};
endmodule

module right_rotate(
  input [3:0] data,
  output [3:0] rotated_data
);
  assign rotated_data = {data[0], data[3:1]};
endmodule
