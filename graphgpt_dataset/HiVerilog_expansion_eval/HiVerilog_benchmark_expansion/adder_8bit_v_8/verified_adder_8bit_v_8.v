module subtractor_8bit(
    input [7:0] a, b,
    input bin,
    output [7:0] diff,
    output bout);

    wire [8:0] borrow;

    full_subtractor FS0 (.a(a[0]), .b(b[0]), .bin(bin), .diff(diff[0]), .bout(borrow[0]));
    full_subtractor FS1 (.a(a[1]), .b(b[1]), .bin(borrow[0]), .diff(diff[1]), .bout(borrow[1]));
    full_subtractor FS2 (.a(a[2]), .b(b[2]), .bin(borrow[1]), .diff(diff[2]), .bout(borrow[2]));
    full_subtractor FS3 (.a(a[3]), .b(b[3]), .bin(borrow[2]), .diff(diff[3]), .bout(borrow[3]));
    full_subtractor FS4 (.a(a[4]), .b(b[4]), .bin(borrow[3]), .diff(diff[4]), .bout(borrow[4]));
    full_subtractor FS5 (.a(a[5]), .b(b[5]), .bin(borrow[4]), .diff(diff[5]), .bout(borrow[5]));
    full_subtractor FS6 (.a(a[6]), .b(b[6]), .bin(borrow[5]), .diff(diff[6]), .bout(borrow[6]));
    full_subtractor FS7 (.a(a[7]), .b(b[7]), .bin(borrow[6]), .diff(diff[7]), .bout(borrow[7]));

    assign bout = borrow[7];
endmodule

module full_subtractor (input a, b, bin, output diff, bout);
    assign {bout, diff} = bin ? (a - b - 1) : (a - b);
endmodule
