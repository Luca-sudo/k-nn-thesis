{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }: 
	flake-utils.lib.eachDefaultSystem(system:
		let
			pkgs = nixpkgs.legacyPackages.${system};
		in{
			devShell = pkgs.mkShell {
				buildInputs = with pkgs.python311Packages; [
					faiss
					numpy
					h5py
          matplotlib
          pandas
          seaborn
          tables
				];
			};
		}
	);
}
