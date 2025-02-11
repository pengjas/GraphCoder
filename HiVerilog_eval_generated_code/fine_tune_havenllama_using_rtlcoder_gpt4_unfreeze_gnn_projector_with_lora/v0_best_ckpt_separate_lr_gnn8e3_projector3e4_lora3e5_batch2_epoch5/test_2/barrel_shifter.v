module barrel_shifter (
    input [7:0] in,
    input [2:0] ctrl,
    output [7:0] out
);

reg [7:0] shifted_in;
reg [7:0] mux_out;



always @(*) begin
    case (ctrl[2])
        4'b000000000: shifted_in <= in;
        4'b00000: shifted_in <= {4'b0, in[3:0]};
        2'b00000: shifted_in <= {2'b0, in[3:0]};
        1'b00000: shifted_in <= {1'b0, in[3:0]};
    endcase
end


always @(*) begin
    case (ctrl[1])
        1'b00000: mux_out <= shifted_in[0];
        1'b00001: mux_out <= shifted_in[1];
        1'b00010: mux_out <= shifted_in[2];
        1'b00100: mux_out <= shifted_in[3];
    endcase
    case (ctrl[0])
        1'b00000: mux_out <= shifted_in[0];
        1'b0001: mux_out <= shifted_in[1];
        1'b0010: mux_out <= shifted_in[2];
        1'b0011: mux_out <= shifted_in[3];
    endcase
end


assign out = mux_out;

endmodule


module barrel_shifter (
    input [7:0] in,
    input [2:0] ctrl,
    output [7:0] out
);

reg [7:0] shifted_in;
reg [7:0] mux_out;


always @(*) begin
    case (ctrl[2])
        4'b00000: shifted_in <= in;
        4'b00001: shifted_in <= {4'b0, in[3:0]};
        2'b00000: shifted_in <= {2'b0, in[3:0]};
        1'b00000: shifted_in <= {1'b0, in[3:0]};
    endcase
end


always @(*) begin
    case (ctrl[1])
        1'b00000: mux_out <= shifted_in[0];
        1'b0001: mux_out <= shifted_in[1];
        1'b00010: mux_out <= shifted_in[2];
        1'b0010: mux_out <= shifted_in[3];
    endcase
    case (ctrl[0])
        1'b00000: mux_out <= shifted_in[0];
        1'b0001: mux_out <= shifted_in[1];
        1'b0010: mux_out <= shifted_in[2];
        1'b0011: mux_out <= shifted_in[3];
    endcase
end


assign out = mux_out;

endmodule