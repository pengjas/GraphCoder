module bitwise_reverse8 (
    input [7:0] din,
    output [7:0] dout
);
    wire [7:0] temp;

    // Instantiating the first bit reversing module
    bit_reverse br1 (
        .data_in(din),
        .data_out(temp)
    );

    // Instantiating the second bit reversing module
    bit_reverse br2 (
        .data_in(temp),
        .data_out(dout)
    );
endmodule

module bit_reverse (
    input [7:0] data_in,
    output reg [7:0] data_out
);
    always @(*) begin
        data_out[0] = data_in[7];
        data_out[1] = data_in[6];
        data_out[2] = data_in[5];
        data_out[3] = data_in[4];
        data_out[4] = data_in[3];
        data_out[5] = data_in[2];
        data_out[6] = data_in[1];
        data_out[7] = data_in[0];
    end
endmodule
