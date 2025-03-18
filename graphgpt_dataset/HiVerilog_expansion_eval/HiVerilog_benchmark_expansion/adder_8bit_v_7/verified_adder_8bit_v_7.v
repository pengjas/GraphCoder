module subtractor_8bit(
    input [7:0] a, b, 
    input bin, 
    output [7:0] diff, 
    output bout);

    wire [8:0] borrow;

    one_bit_subtractor S0 (.a(a[0]), .b(b[0]), .bin(bin), .diff(diff[0]), .bout(borrow[0]));
    one_bit_subtractor S1 (.a(a[1]), .b(b[1]), .bin(borrow[0]), .diff(diff[1]), .bout(borrow[1]));
    one_bit_subtractor S2 (.a(a[2]), .b(b[2]), .bin(borrow[1]), .diff(diff[2]), .bout(borrow[2]));
    one_bit_subtractor S3 (.a(a[3]), .b(b[3]), .bin(borrow[2]), .diff(diff[3]), .bout(borrow[3]));
    one_bit_subtractor S4 (.a(a[4]), .b(b[4]), .bin(borrow[3]), .diff(diff[4]), .bout(borrow[4]));
    one_bit_subtractor S5 (.a(a[5]), .b(b[5]), .bin(borrow[4]), .diff(diff[5]), .bout(borrow[5]));
    one_bit_subtractor S6 (.a(a[6]), .b(b[6]), .bin(borrow[5]), .diff(diff[6]), .bout(borrow[6]));
    one_bit_subtractor S7 (.a(a[7]), .b(b[7]), .bin(borrow[6]), .diff(diff[7]), .bout(borrow[7]));

    assign bout = borrow[7]; 
endmodule

module one_bit_subtractor (input a, b, bin, output diff, bout);
    assign {bout, diff} = a - b - bin;
endmodule